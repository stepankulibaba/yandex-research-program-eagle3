"""Task 5 input: how long the target takes to regenerate the paper's training dialogues (ShareGPT + UltraChat).

    python regen_probe.py <server url> <model dir> <out.json> [--samples-per-bucket 64] [--scan-limit 8192]
                          [--concurrency 64]

The paper rewrites every assistant turn of ShareGPT (~68K) and UltraChat-200K (~464K) with the target. Here:
  1. scan the first `scan-limit` dialogues of each source, split them by length (0-512 / 513-1024 / 1025+ tokens)
     and draw `samples-per-bucket` dialogues per group (reservoir sampling, seed 0); the group sizes give weights;
  2. regenerate the drawn dialogues turn by turn on a plain SGLang server, many at a time (like a data job),
     with the training policy: the authors' system prompt, at most 512 new tokens per turn, at most 1900 tokens
     per training example;
  3. per group: seconds per dialogue, training tokens per dialogue, share of dialogues the policy keeps.
budget.py scales these to the full corpora. Finished dialogues are journaled, so a restart continues.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import time

from common import MODELS, SCHEMA, SYSTEM, atomic_json, digest, positive, read_json
from sglang_client import clean_text, stop_ids

# name: (Hugging Face dataset, pinned revision, splits, dialogue count assumed by the paper)
SOURCES = {
    'sharegpt': ('Aeala/ShareGPT_Vicuna_unfiltered', '8b0048ad6ae8c22f46a78c15559dec98feef5539', ['train'], 68000),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', '8049631c405ae6576f93f445c6b8166f76f5505a',
                  ['train_sft', 'train_gen'], 464000),
}
BUCKETS = ('0-512', '513-1024', '1025+')
POLICY = {'max_training_tokens': 1900, 'max_new_tokens_per_turn': 512, 'system': SYSTEM,
          'temperature': 0.0, 'seed': 0}


def normalize(item):
    """A dialogue as [{'role': 'user'|'assistant', 'content': ...}, ...] alternating from the user, or None."""
    messages = []
    for m in item.get('messages', item.get('conversations', [])):
        role = m.get('role', m.get('from'))
        role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
        if role == 'system':
            continue
        text = m.get('content', m.get('value'))
        if role not in ('user', 'assistant') or not isinstance(text, str) or not text.strip():
            return None
        messages.append({'role': role, 'content': text})
    alternating = ['user', 'assistant'] * (len(messages) // 2)
    if len(messages) < 2 or len(messages) % 2 or [m['role'] for m in messages] != alternating:
        return None
    return messages


def length_bucket(tokens):
    return '0-512' if tokens <= 512 else ('513-1024' if tokens <= 1024 else '1025+')


def sample_source(tok, spec, per_bucket, scan_limit):
    """Step 1: reservoir sample of `per_bucket` dialogues per length group from the first `scan_limit` rows."""
    from datasets import load_dataset
    repo, revision, splits, _ = spec
    rng = random.Random(0)
    groups = {b: [] for b in BUCKETS}
    counts = {b: 0 for b in BUCKETS}
    scanned = 0
    for split in splits:
        stream = load_dataset(repo, revision=revision, split=split, streaming=True)
        for i, item in enumerate(stream):
            if i >= scan_limit:
                break
            scanned += 1
            dialogue = normalize(item)
            if dialogue is None:
                continue
            length = len(tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}] + dialogue,
                                                 tokenize=True, add_generation_prompt=False))
            bucket = length_bucket(length)
            counts[bucket] += 1
            row = {'id': digest([repo, revision, split, i, dialogue]), 'messages': dialogue,
                   'original_tokens': length}
            pool = groups[bucket]
            if len(pool) < per_bucket:
                pool.append(row)
            else:
                j = rng.randrange(counts[bucket])
                if j < per_bucket:
                    pool[j] = row
    if not sum(counts.values()):
        raise ValueError('No valid source dialogues')
    return {'groups': groups, 'population_counts': counts, 'scanned': scanned, 'valid': sum(counts.values())}


def regenerate(server, tok, row, stops):
    """Step 2 for one dialogue: replace every assistant turn with the target's own answer."""
    import requests
    messages = [{'role': 'system', 'content': SYSTEM}]
    out_tokens, turns, seconds, capped = 0, 0, 0.0, False
    for user in row['messages'][::2]:
        candidate = messages + [user]
        ids = tok.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True)
        room = POLICY['max_training_tokens'] - len(ids) - 16
        if room <= 0:                   # the dialogue no longer fits a training example
            capped = True
            break
        limit = min(POLICY['max_new_tokens_per_turn'], room)
        params = {'temperature': 0.0, 'max_new_tokens': limit, 'stop_token_ids': stops,
                  'top_p': 1.0, 'top_k': -1, 'sampling_seed': 0}
        t0 = time.perf_counter()
        response = requests.post(server + '/generate', json={'input_ids': ids, 'sampling_params': params},
                                 timeout=(10, 600))
        response.raise_for_status()
        seconds += time.perf_counter() - t0
        out = response.json()
        meta = out['meta_info']
        if not meta.get('finish_reason') or meta['completion_tokens'] > limit:
            raise ValueError('Invalid regenerated answer')
        capped |= isinstance(meta['finish_reason'], dict) and meta['finish_reason'].get('type') == 'length'
        out_tokens += meta['completion_tokens']
        messages = candidate + [{'role': 'assistant', 'content': clean_text(out['text'], tok)}]
        turns += 1
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    accepted = turns > 0 and len(ids) <= POLICY['max_training_tokens']
    return {'id': row['id'], 'output_tokens': out_tokens, 'assistant_turns': turns,
            'training_tokens': len(ids) if accepted else 0, 'accepted': accepted,
            'capped': capped, 'request_seconds': seconds}


def validate_probe(path):
    r = read_json(path)
    if r.get('schema') != SCHEMA or r.get('complete') is not True or r.get('policy') != POLICY:
        raise ValueError('Incomplete or incompatible regeneration pilot')
    if set(r['sources']) != set(SOURCES):
        raise ValueError('Missing source dataset')
    for source in r['sources'].values():
        weights = []
        for g in source['groups'].values():
            positive(g['weight'], 'weight')
            positive(g['samples'], 'samples')
            positive(g['seconds_per_dialogue'], 'seconds per dialogue')
            positive(g['mean_training_tokens'], 'training tokens', zero=True)
            weights.append(g['weight'])
        if abs(sum(weights) - 1) > 1e-8:
            raise ValueError('Incomplete stratum coverage')
    return r


def regenerate_group(args, tok, stops, rows, journal_path):
    """Step 2 for one length group, many dialogues at a time; returns (results, wall seconds incl. earlier runs)."""
    try:
        journal = read_json(journal_path)
    except (OSError, ValueError):
        journal = {'rows': {}, 'wall_seconds': 0.0}
    pending = [r for r in rows if r['id'] not in journal['rows']]
    earlier_wall, t0 = journal['wall_seconds'], time.perf_counter()
    errors = []
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(regenerate, args.server, tok, row, stops) for row in pending]
        for future in as_completed(futures):
            try:
                result = future.result()
                journal['rows'][result['id']] = result
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
            journal['wall_seconds'] = earlier_wall + time.perf_counter() - t0
            atomic_json(journal_path, journal)
    if errors:
        raise RuntimeError('Regeneration failed; successful requests preserved: ' + errors[0])
    if len(journal['rows']) != len({r['id'] for r in rows}):
        raise ValueError('Incomplete regeneration journal')
    return list(journal['rows'].values()), journal['wall_seconds']


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('model')
    parser.add_argument('output')
    parser.add_argument('--samples-per-bucket', type=int, default=64)
    parser.add_argument('--scan-limit', type=int, default=8192)
    parser.add_argument('--concurrency', type=int, default=64)
    args = parser.parse_args()
    if min(args.samples_per_bucket, args.scan_limit, args.concurrency) <= 0:
        raise ValueError('Positive sample/scan/concurrency counts required')
    tok = AutoTokenizer.from_pretrained(args.model)
    stops = stop_ids(tok, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'sources': SOURCES, 'model': MODELS['target'], 'policy': POLICY,
                  'samples_per_bucket': args.samples_per_bucket, 'scan_limit': args.scan_limit}

    # Step 1, cached: the same draw on a restart.
    selection_path = output.with_suffix('.selection.json')
    try:
        selection = read_json(selection_path)
        if selection['fingerprint'] != digest(provenance):
            raise ValueError('Stale selection')
    except (OSError, ValueError, KeyError):
        selection = {'fingerprint': digest(provenance), 'sources': {
            name: sample_source(tok, spec, args.samples_per_bucket, args.scan_limit)
            for name, spec in SOURCES.items()}}
        atomic_json(selection_path, selection)

    # Steps 2-3.
    result = {'schema': SCHEMA, 'complete': True, 'policy': POLICY, 'provenance': provenance,
              'sampling': 'reservoir within bounded prefixes of the named splits; preliminary weights',
              'concurrency': args.concurrency, 'sources': {}}
    for name, spec in SOURCES.items():
        selected = selection['sources'][name]
        groups = {}
        for bucket, rows in selected['groups'].items():
            population = selected['population_counts'][bucket]
            if not population:
                continue
            if not rows:
                raise ValueError('Nonempty stratum without samples')
            key = digest({'provenance': provenance, 'source': name, 'bucket': bucket,
                          'ids': [r['id'] for r in rows], 'concurrency': args.concurrency})
            journal_path = output.parent / 'regen_journal' / f"{name}_{bucket.replace('+', 'plus')}_{key}.json"
            done, wall_seconds = regenerate_group(args, tok, stops, rows, journal_path)
            n = len(done)
            groups[bucket] = {'samples': n, 'weight': population / selected['valid'],
                              'seconds_per_dialogue': wall_seconds / n,      # wall time shared by the batch
                              'mean_training_tokens': sum(r['training_tokens'] for r in done) / n,
                              'accepted_fraction': sum(r['accepted'] for r in done) / n,
                              'mean_assistant_turns': sum(r['assistant_turns'] for r in done) / n,
                              'capped_fraction': sum(r['capped'] for r in done) / n,
                              'effective_concurrency': min(args.concurrency, len(rows))}
        result['sources'][name] = {'repo': spec[0], 'revision': spec[1], 'splits': spec[2],
                                   'paper_assumed_dialogues': spec[3], 'scanned': selected['scanned'],
                                   'valid': selected['valid'], 'groups': groups}
    atomic_json(output, result)
    validate_probe(output)
    print(json.dumps(result, indent=2))
    # Streaming `datasets` leaves background threads that crash CPython's shutdown ("PyGILState_Release"),
    # turning a finished run into a failure. Everything is written above, so skip the shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
