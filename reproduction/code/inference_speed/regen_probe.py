"""Task 5 input: how long the target takes to regenerate the training dialogues (ShareGPT + UltraChat).

    python regen_probe.py <server url> <model dir> <out.json> [--samples-per-bucket 64] [--scan-limit 8192]
                          [--concurrency 64]

The paper says only that the target model generates the responses (no lengths, turns or temperature are given),
so the policy below is OURS: every assistant turn is rewritten, greedily, with the authors' system prompt, at most
512 new tokens per turn and at most 1900 tokens per training example. A dialogue that does not fit is dropped
whole, as the authors' trainer drops examples longer than its max_len (traineagle3/main.py).

Each split is its own stratum, weighted by its real size:
  1. scan the first `scan-limit` dialogues of the split, sort them into length groups (0-512 / 513-1024 / 1025+
     tokens) and draw `samples-per-bucket` per group (reservoir sampling, seed 0); group shares give weights;
  2. regenerate the drawn dialogues on a plain SGLang server, many at a time (like a data job);
  3. per group: seconds per dialogue, kept training tokens per dialogue, share of dialogues kept.
budget.py scales these to the full splits. Finished dialogues are journaled, so a restart continues.
ShareGPT is ShareGPT_V4.3_unfiltered_cleaned_split.json, the file the EAGLE authors use: 68 623 dialogues, the
paper's "~68K". Limit of the pilot: only the start of each split is scanned.
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

# name: (Hugging Face dataset, pinned revision, {split: dialogues it stands for}, file to read or None)
# ShareGPT: only the file the EAGLE authors use; the repo also holds another file with a different schema,
# and `datasets` would merge the two.
SOURCES = {
    'sharegpt': ('Aeala/ShareGPT_Vicuna_unfiltered', '8b0048ad6ae8c22f46a78c15559dec98feef5539',
                 {'train': 68623}, 'ShareGPT_V4.3_unfiltered_cleaned_split.json'),     # all of it: the paper's ~68K
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', '8049631c405ae6576f93f445c6b8166f76f5505a',
                  {'train_sft': 207865, 'train_gen': 256032}, None),   # real sizes; together the paper's ~464K
}
BUCKETS = ('0-512', '513-1024', '1025+')
POLICY = {'max_training_tokens': 1900, 'max_new_tokens_per_turn': 512, 'system': SYSTEM,
          'temperature': 0.0, 'seed': 0, 'too_long': 'drop the dialogue'}


def normalize(item):
    """A dialogue as [{'role': 'user'|'assistant', 'content': ...}, ...] alternating from the user, or None."""
    messages = []
    source = item.get('messages', item.get('conversations', []))
    if isinstance(source, str):       # a dialogue stored as a JSON string
        try:
            source = json.loads(source)
        except ValueError:
            return None
    if isinstance(source, dict):      # `datasets` returns a sequence of records as columns: {'from': [...], ...}
        keys = list(source)
        lengths = {len(source[k]) for k in keys if isinstance(source[k], list)}
        if not keys or len(lengths) != 1:
            return None
        source = [{k: source[k][i] for k in keys} for i in range(lengths.pop())]
    if not isinstance(source, list):
        return None
    for m in source:
        if not isinstance(m, dict):      # some ShareGPT rows hold plain strings: not a usable dialogue
            return None
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


def sample_split(tok, repo, revision, split, per_bucket, scan_limit, data_file=None):
    """Step 1 for one split: reservoir sample of `per_bucket` dialogues per length group."""
    rng = random.Random(0)
    groups = {b: [] for b in BUCKETS}
    counts = {b: 0 for b in BUCKETS}
    scanned = 0
    if data_file:
        # A plain JSON file (ShareGPT): read it as is. `datasets` streaming mangles its nested records.
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, data_file, repo_type='dataset', revision=revision)
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
    else:
        from datasets import load_dataset
        rows = load_dataset(repo, revision=revision, split=split, streaming=True)
    for i, item in enumerate(rows):
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
        row = {'id': digest([repo, revision, split, i, dialogue]), 'messages': dialogue, 'original_tokens': length}
        pool = groups[bucket]
        if len(pool) < per_bucket:
            pool.append(row)
        else:
            j = rng.randrange(counts[bucket])
            if j < per_bucket:
                pool[j] = row
    if not sum(counts.values()):
        raise ValueError(f'No valid dialogues in {repo}:{split}')
    return {'groups': groups, 'population_counts': counts, 'scanned': scanned, 'valid': sum(counts.values())}


def regenerate(server, tok, row, stops):
    """Step 2 for one dialogue: replace every assistant turn with the target's own answer."""
    import requests
    limit_per_turn = POLICY['max_new_tokens_per_turn']
    messages = [{'role': 'system', 'content': SYSTEM}]
    out_tokens, turns, seconds, too_long = 0, 0, 0.0, False
    for user in row['messages'][::2]:
        candidate = messages + [user]
        ids = tok.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True)
        room = POLICY['max_training_tokens'] - len(ids) - 16
        if room <= 0:                    # the next turn no longer fits a training example
            too_long = True
            break
        limit = min(limit_per_turn, room)
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
        out_tokens += meta['completion_tokens']
        turns += 1
        cut = isinstance(meta['finish_reason'], dict) and meta['finish_reason'].get('type') == 'length'
        if cut and limit < limit_per_turn:   # the answer was cut only to fit the example: it does not fit
            too_long = True
            break
        messages = candidate + [{'role': 'assistant', 'content': clean_text(out['text'], tok)}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    kept = turns > 0 and not too_long and len(ids) <= POLICY['max_training_tokens']
    return {'id': row['id'], 'output_tokens': out_tokens, 'assistant_turns': turns,
            'training_tokens': len(ids) if kept else 0, 'accepted': kept,
            'too_long': too_long, 'request_seconds': seconds}


def validate_probe(path):
    r = read_json(path)
    if r.get('schema') != SCHEMA or r.get('complete') is not True or r.get('policy') != POLICY:
        raise ValueError('Incomplete or incompatible regeneration pilot')
    if set(r['sources']) != set(SOURCES):
        raise ValueError('Missing source dataset')
    for name, source in r['sources'].items():
        if set(source['splits']) != set(SOURCES[name][2]):
            raise ValueError('Missing split')
        for split in source['splits'].values():
            positive(split['dialogues'], 'dialogues')
            weights = []
            for g in split['groups'].values():
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
    stops = stop_ids(tok)
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
            name: {split: sample_split(tok, repo, revision, split, args.samples_per_bucket, args.scan_limit,
                                       data_file)
                   for split in splits}
            for name, (repo, revision, splits, data_file) in SOURCES.items()}}
        atomic_json(selection_path, selection)

    # Steps 2-3.
    result = {'schema': SCHEMA, 'complete': True, 'policy': POLICY, 'provenance': provenance,
              'sampling': 'per split: reservoir within the first scan-limit rows; preliminary weights',
              'concurrency': args.concurrency, 'sources': {}}
    for name, (repo, revision, splits, _) in SOURCES.items():
        result['sources'][name] = {'repo': repo, 'revision': revision, 'splits': {}}
        for split, dialogues in splits.items():
            selected = selection['sources'][name][split]
            groups = {}
            for bucket, rows in selected['groups'].items():
                population = selected['population_counts'][bucket]
                if not population:
                    continue
                if not rows:
                    raise ValueError('Nonempty stratum without samples')
                key = digest({'provenance': provenance, 'source': name, 'split': split, 'bucket': bucket,
                              'ids': [r['id'] for r in rows], 'concurrency': args.concurrency})
                journal_path = (output.parent / 'regen_journal'
                                / f"{name}_{split}_{bucket.replace('+', 'plus')}_{key}.json")
                done, wall_seconds = regenerate_group(args, tok, stops, rows, journal_path)
                n = len(done)
                groups[bucket] = {'samples': n, 'weight': population / selected['valid'],
                                  'seconds_per_dialogue': wall_seconds / n,     # wall time shared by the batch
                                  'mean_training_tokens': sum(r['training_tokens'] for r in done) / n,
                                  'accepted_fraction': sum(r['accepted'] for r in done) / n,
                                  'too_long_fraction': sum(r['too_long'] for r in done) / n,
                                  'mean_assistant_turns': sum(r['assistant_turns'] for r in done) / n,
                                  'effective_concurrency': min(args.concurrency, len(rows))}
            result['sources'][name]['splits'][split] = {
                'dialogues': dialogues, 'scanned': selected['scanned'], 'valid': selected['valid'], 'groups': groups}
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
