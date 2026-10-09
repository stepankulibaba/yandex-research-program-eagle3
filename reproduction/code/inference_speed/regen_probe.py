"""Stratified, resumable multi-turn regeneration pilot under the training policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import json
from pathlib import Path
import random
import os
import sys
import time

from runtime import (MODELS, SCHEMA, SYSTEM, atomic_json, digest, positive, read_json)
from sglang_client import clean_text, stop_ids

SOURCES = {
    'sharegpt': ('Aeala/ShareGPT_Vicuna_unfiltered', '8b0048ad6ae8c22f46a78c15559dec98feef5539', ['train'], 68000),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', '8049631c405ae6576f93f445c6b8166f76f5505a', ['train_sft', 'train_gen'], 464000),
}
BUCKETS = ('0-512', '513-1024', '1025+')
POLICY = {'max_training_tokens': 1900, 'max_new_tokens_per_turn': 512, 'system': SYSTEM,
          'temperature': 0.0, 'seed': 0}


def normalize(item):
    source = item.get('messages', item.get('conversations', []))
    result = []
    for m in source:
        role = m.get('role', m.get('from'))
        role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
        if role == 'system':
            continue
        text = m.get('content', m.get('value'))
        if role not in ('user', 'assistant') or not isinstance(text, str) or not text.strip():
            return None
        result.append({'role': role, 'content': text})
    if len(result) < 2 or len(result) % 2 or [m['role'] for m in result] != ['user', 'assistant'] * (len(result) // 2):
        return None
    return result


def sample_source(tok, spec, n, scan_limit):
    from datasets import load_dataset
    repo, revision, splits, _ = spec
    rng = random.Random(0)
    groups = {b: [] for b in BUCKETS}
    counts = {b: 0 for b in BUCKETS}
    scanned = 0
    for split in splits:
        stream = load_dataset(repo, revision=revision, split=split, streaming=True)
        # A bounded reservoir is explicitly a pilot, not a uniform full-corpus estimate.
        for i, item in enumerate(stream):
            if i >= scan_limit:
                break
            scanned += 1
            conv = normalize(item)
            if conv is None:
                continue
            length = len(tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}] + conv,
                                                 tokenize=True, add_generation_prompt=False))
            bucket = '0-512' if length <= 512 else ('513-1024' if length <= 1024 else '1025+')
            counts[bucket] += 1
            row = {'id': digest([repo, revision, split, i, conv]), 'messages': conv, 'original_tokens': length}
            pool = groups[bucket]
            if len(pool) < n:
                pool.append(row)
            else:
                j = rng.randrange(counts[bucket])
                if j < n:
                    pool[j] = row
    if not sum(counts.values()):
        raise ValueError('No valid source dialogues')
    return {'groups': groups, 'population_counts': counts, 'scanned': scanned, 'valid': sum(counts.values())}


def regenerate(server, tok, row, stops):
    import requests
    messages = [{'role': 'system', 'content': SYSTEM}]
    out_tokens, turns, seconds = 0, 0, 0.0
    capped = False
    for user in row['messages'][::2]:
        candidate = messages + [user]
        ids = tok.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True)
        remaining = POLICY['max_training_tokens'] - len(ids) - 16
        if remaining <= 0:
            capped = True
            break
        limit = min(POLICY['max_new_tokens_per_turn'], remaining)
        params = {'temperature': 0.0, 'max_new_tokens': limit, 'stop_token_ids': stops,
                  'top_p': 1.0, 'top_k': -1, 'sampling_seed': 0}
        t0 = time.perf_counter()
        response = requests.post(server + '/generate', json={'input_ids': ids, 'sampling_params': params}, timeout=(10, 600))
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
    selection_path = output.with_suffix('.selection.json')
    try:
        selection = read_json(selection_path)
        if selection['fingerprint'] != digest(provenance):
            raise ValueError('Stale selection')
    except (OSError, ValueError, KeyError):
        selection = {'fingerprint': digest(provenance), 'sources': {
            name: sample_source(tok, spec, args.samples_per_bucket, args.scan_limit) for name, spec in SOURCES.items()}}
        atomic_json(selection_path, selection)
    result = {'schema': SCHEMA, 'complete': True, 'policy': POLICY, 'provenance': provenance,
              'sampling': 'reservoir within bounded prefixes of the named splits; preliminary weights',
              'concurrency': args.concurrency, 'sources': {}}
    for name, spec in SOURCES.items():
        selected = selection['sources'][name]
        groups = {}
        for bucket, rows in selected['groups'].items():
            count = selected['population_counts'][bucket]
            if not count:
                continue
            if not rows:
                raise ValueError('Nonempty stratum without samples')
            key = digest({'provenance': provenance, 'source': name, 'bucket': bucket,
                          'ids': [r['id'] for r in rows], 'concurrency': args.concurrency})
            cache_path = output.parent / 'regen_journal' / (name + '_' + bucket.replace('+', 'plus') + '_' + key + '.json')
            try:
                cache = read_json(cache_path)
            except (OSError, ValueError):
                cache = {'rows': {}, 'wall_seconds': 0.0}
            pending = [r for r in rows if r['id'] not in cache['rows']]
            old_wall, t0 = cache['wall_seconds'], time.perf_counter()
            errors = []
            with ThreadPoolExecutor(args.concurrency) as pool:
                futures = {pool.submit(regenerate, args.server, tok, row, stops): row for row in pending}
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        cache['rows'][r['id']] = r
                    except Exception as exc:
                        errors.append(f'{type(exc).__name__}: {exc}')
                    cache['wall_seconds'] = old_wall + time.perf_counter() - t0
                    atomic_json(cache_path, cache)
            if errors:
                raise RuntimeError('Regeneration failed; successful requests preserved: ' + errors[0])
            if len(cache['rows']) != len({r['id'] for r in rows}):
                raise ValueError('Incomplete regeneration journal')
            values = list(cache['rows'].values())
            n = len(values)
            # Includes amortized measured retry/tail cost; restored groups retain their wall time.
            groups[bucket] = {'samples': n, 'weight': count / selected['valid'],
                'seconds_per_dialogue': cache['wall_seconds'] / n,
                'mean_training_tokens': sum(r['training_tokens'] for r in values) / n,
                'accepted_fraction': sum(r['accepted'] for r in values) / n,
                'mean_assistant_turns': sum(r['assistant_turns'] for r in values) / n,
                'capped_fraction': sum(r['capped'] for r in values) / n,
                'effective_concurrency': min(args.concurrency, len(rows))}
        result['sources'][name] = {'repo': spec[0], 'revision': spec[1], 'splits': spec[2],
                                  'paper_assumed_dialogues': spec[3], 'scanned': selected['scanned'],
                                  'valid': selected['valid'], 'groups': groups}
    atomic_json(output, result)
    validate_probe(output)
    print(json.dumps(result, indent=2))
    # Streaming `datasets` leaves background threads that crash CPython's finalization
    # ("PyGILState_Release ... must be current"), turning a finished run into exit code 134.
    # Every output is written and fsynced above, so leave without finalization.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
