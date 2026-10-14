"""Training data: the whole regenerated PerfectBlend, tokenized by the authors' own preprocessing (CPU only).

    python prepare_data.py [--limit N] [--procs 32]        (--limit: first N dialogues, for the smoke test)

Each dialogue goes through build_dataset_rank.preprocess_function taken from the authors' traineagle3/main.py and
run as is (their system prompt, the Llama-3 chat template, their loss mask), so token ids and masks are exactly what
their trainer would see. Like them we drop (not truncate) examples longer than 2048 tokens.

Known quirk of their loss mask, kept as is: in multi-turn dialogues the last two positions of every reply except
the last one are not supervised (~0.2 % of supervised tokens on PerfectBlend). Both variants see the same data.

Writes data/ (or data_smoke/ with --limit):
    train/, val/       datasets.save_to_disk: input_ids, loss_mask (attention mask is all ones)
    vocab.json         the 32K draft vocabulary: the most frequent supervised tokens of train/ (as the authors'
                       scandata, which counts tokens under the loss mask), sorted target ids
    stats.json         counts, dropped dialogues by reason, length histogram, hashes
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from settings import DATA, DATASET, DATASET_REVISION, EAGLE_REPO, MAX_LEN, SEED, TARGET, VAL_DIALOGUES

DRAFT_VOCAB = 32000


def author_preprocessor(main_py, tokenizer):
    """build_dataset_rank.preprocess_function from the authors' main.py, compiled on its own."""
    import torch
    tree = ast.parse(Path(main_py).read_text(encoding='utf-8'))
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dataset_rank')
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_function')
    namespace = {'torch': torch, 'tokenizer': tokenizer, 'train_config': {'max_len': MAX_LEN}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[])), str(main_py), 'exec'),
         namespace)
    return namespace['preprocess_function']


def to_sharegpt(dialogue):
    """PerfectBlend {role, content} turns -> the authors' {from, value}; None if their code would choke on it."""
    if not isinstance(dialogue, list) or len(dialogue) < 2:
        return None
    roles = [m.get('role') if isinstance(m, dict) else None for m in dialogue]
    if roles[0] == 'assistant':                 # the authors skip a leading model turn; do the same up front
        dialogue, roles = dialogue[1:], roles[1:]
    if not roles or roles != ['user', 'assistant'] * (len(roles) // 2) + ['user'] * (len(roles) % 2):
        return None                             # their assert would kill the whole map() worker
    if 'assistant' not in roles or not all(isinstance(m.get('content'), str) and m['content'].strip()
                                           for m in dialogue):
        return None
    return [{'from': 'human' if r == 'user' else 'gpt', 'value': m['content']} for r, m in zip(roles, dialogue)]


def tokenize_batch(batch, tokenizer_dir, main_py):
    """datasets.map worker: raw rows -> token ids and loss masks (empty lists mark a dropped row and its reason)."""
    from transformers import AutoTokenizer
    global _PREPROCESS
    if '_PREPROCESS' not in globals():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer.pad_token_id = 0
        _PREPROCESS = author_preprocessor(main_py, tokenizer)
    out = {'id': [], 'input_ids': [], 'loss_mask': [], 'drop': []}
    for rid, dialogue in zip(batch['id'], batch['conversations']):
        sharegpt = to_sharegpt(dialogue)
        ids = mask = []
        reason = ''
        if sharegpt is None:
            reason = 'format'
        else:
            pre = _PREPROCESS({'id': [str(rid)], 'conversations': [sharegpt]})
            if not pre['input_ids']:
                reason = 'too_long'
            else:
                ids = pre['input_ids'][0][0].tolist()
                mask = pre['loss_mask'][0][0].tolist()
                if not sum(mask):
                    ids, mask, reason = [], [], 'no_supervised_tokens'
        out['id'].append(str(rid))
        out['input_ids'].append(ids)
        out['loss_mask'].append(mask)
        out['drop'].append(reason)
    return out


def corpus_pass(train):
    """One streaming pass (numpy, batches): supervised-token counts, lengths, totals."""
    import numpy as np
    vocab_size = json.loads((TARGET / 'config.json').read_text(encoding='utf-8'))['vocab_size']
    counts = np.zeros(vocab_size, dtype=np.int64)
    lengths = Counter()
    tokens = supervised = 0
    for batch in train.with_format('numpy').iter(batch_size=20000):
        picked = []
        for ids, mask in zip(batch['input_ids'], batch['loss_mask']):
            ids, mask = np.asarray(ids), np.asarray(mask)
            picked.append(ids[mask == 1])
            lengths[min(len(ids) // 512, 3)] += 1
            tokens += len(ids)
        picked = np.concatenate(picked)
        supervised += len(picked)
        counts += np.bincount(picked, minlength=vocab_size)
    return counts, lengths, tokens, supervised


def draft_vocabulary(counts):
    """Top-32K supervised tokens (ties by token id), sorted target ids; as the authors' scandata."""
    import numpy as np
    order = np.lexsort((np.arange(len(counts)), -counts))      # by count desc, then id asc
    chosen = order[:DRAFT_VOCAB]
    coverage = float(counts[chosen].sum() / max(1, counts.sum()))
    return sorted(int(t) for t in chosen), coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--procs', type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    from datasets import load_dataset
    out = DATA if not args.limit else DATA.parent / 'data_smoke'
    out.mkdir(parents=True, exist_ok=True)

    if args.limit:      # smoke test: stream the first rows instead of downloading all 1.8 GB
        from datasets import Dataset
        stream = load_dataset(DATASET, revision=DATASET_REVISION, split='train', streaming=True)
        raw = Dataset.from_list(list(stream.take(args.limit)))
    else:
        raw = load_dataset(DATASET, revision=DATASET_REVISION, split='train')
    tokenized = raw.map(tokenize_batch, batched=True, batch_size=256, num_proc=args.procs,
                        remove_columns=raw.column_names,
                        fn_kwargs={'tokenizer_dir': str(TARGET),
                                   'main_py': str(EAGLE_REPO / 'eagle/traineagle3/main.py')})
    drops = Counter(tokenized['drop'])
    drops.pop('', None)                       # '' = kept
    kept = tokenized.filter(lambda r: not r['drop'], num_proc=args.procs).remove_columns('drop')

    # held-out dialogues: a fixed random choice (seeded), never trained on
    n_val = min(VAL_DIALOGUES, len(kept) // 10)
    order = kept.shuffle(seed=SEED + 1)
    val, train = order.select(range(n_val)), order.select(range(n_val, len(order)))
    train.save_to_disk(str(out / 'train'))     # train.py shuffles per epoch with its own seed
    val.save_to_disk(str(out / 'val'))

    counts, lengths, tokens, supervised = corpus_pass(train)
    vocab, coverage = draft_vocabulary(counts)
    (out / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    ids_hash = hashlib.sha256('\n'.join(str(i) for i in train['id']).encode()).hexdigest()
    stats = {'dataset': DATASET, 'revision': DATASET_REVISION, 'limit': args.limit,
             'dialogues_read': len(raw), 'kept': len(kept), 'dropped': dict(drops),
             'train': len(train), 'val': len(val),
             'train_tokens': tokens, 'train_supervised_tokens': supervised,
             'length_histogram_512': {f'{512 * k}-{512 * (k + 1)}' if k < 3 else '1536-2048': v
                                      for k, v in sorted(lengths.items())},
             'draft_vocab_coverage': coverage, 'train_ids_sha256': ids_hash,
             'vocab_sha256': hashlib.sha256(json.dumps(vocab).encode()).hexdigest(),
             'preprocessor': 'traineagle3/main.py build_dataset_rank.preprocess_function (pinned EAGLE)'}
    (out / 'stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))
    sys.stdout.flush()
    os._exit(0)      # datasets' worker threads can crash CPython's shutdown after everything is written


if __name__ == '__main__':
    main()
