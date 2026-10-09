"""Produce ONE author-tokenized corpus/mask/vocab mapping for both speed adapters."""
import ast
from collections import Counter
import json
from pathlib import Path
import os
import sys

PACKAGE = Path(__file__).resolve().parent.parent / 'inference_speed'
sys.path.insert(0, str(PACKAGE))
from runtime import SYSTEM, atomic_json, atomic_text, digest, file_hash

MAX_TOKENS = 1900
SEQ_LENGTH = 2048


def author_preprocessor(source, tok):
    # Execute only the authors' pure preprocessing function, not their trainer.
    import torch
    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dataset_rank')
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_function')
    namespace = {'torch': torch, 'tokenizer': tok, 'train_config': {'max_len': SEQ_LENGTH}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['preprocess_function']


def selected_vocab(rows, vocab_size, size=32000):
    counts = Counter(t for r in rows for t, m in zip(r['input_ids'], r['loss_mask']) if m)
    selected = [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:size]]
    used = set(selected)
    selected.extend(i for i in range(vocab_size) if i not in used)
    selected = sorted(selected[:size])
    if len(selected) != size or min(selected) < 0 or max(selected) >= vocab_size:
        raise ValueError('Invalid shared draft vocabulary')
    return selected


def main(model_dir, n_samples='1000', output='data', eagle_repo=None):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    n_samples = int(n_samples)
    if n_samples <= 0 or n_samples % 2:
        raise ValueError('Use a positive, even sample count for accumulation=2')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(eagle_repo) if eagle_repo else PACKAGE / 'EAGLE'
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token_id = 0
    preprocess = author_preprocessor(repo / 'eagle/traineagle3/main.py', tok)
    source = 'frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct'
    # Revision is resolved once by the orchestrator and included in its manifest.
    revision = __import__('os').environ['TRAIN_DATA_REVISION']
    rows, messages, originals = [], [], []
    for item in load_dataset(source, revision=revision, split='train', streaming=True):
        conv = item['conversations']
        if not isinstance(conv, list) or len(conv) < 2 or len(conv) % 2:
            continue
        if [m['role'] for m in conv] != ['user', 'assistant'] * (len(conv) // 2):
            continue
        if not all(isinstance(m.get('content'), str) and m['content'].strip() for m in conv):
            continue
        rid = str(item['id'])
        original = {'id': rid, 'conversations': [
            {'from': 'human' if m['role'] == 'user' else 'gpt', 'value': m['content']} for m in conv]}
        pre = preprocess({'id': [rid], 'conversations': [original['conversations']]})
        if not pre['input_ids']:
            continue
        ids = pre['input_ids'][0].tolist()[0]
        mask = pre['loss_mask'][0].tolist()[0]
        if not 0 < len(ids) <= MAX_TOKENS or not sum(mask):
            continue
        row = {'id': rid, 'input_ids': ids, 'attention_mask': [1] * len(ids), 'loss_mask': mask}
        # Same chat input as the authors; re-tokenization must agree before saving.
        msgs = [{'role': 'system', 'content': SYSTEM}] + conv
        if tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False) != ids:
            raise ValueError('Chat template parity failure')
        rows.append(row)
        originals.append(original)
        messages.append({'id': rid, 'messages': msgs})
        if len(rows) == n_samples + 16:
            break
    if len(rows) != n_samples + 16:
        raise ValueError(f'Not enough valid examples: {len(rows)}')
    train, test = rows[:n_samples], rows[n_samples:]
    for name, data in [('canonical.jsonl', train), ('test_canonical.jsonl', test),
                       ('sharegpt.jsonl', originals[:n_samples]), ('messages.jsonl', messages[:n_samples])]:
        atomic_text(out / name, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    config = json.loads((Path(model_dir) / 'config.json').read_text(encoding='utf-8'))
    selected = selected_vocab(train, config['vocab_size'])
    atomic_json(out / 'selected_tokens.json', selected)
    hist = {'0-512': 0, '513-1024': 0, '1025-1900': 0}
    for r in train:
        length = len(r['input_ids'])
        hist['0-512' if length <= 512 else ('513-1024' if length <= 1024 else '1025-1900')] += 1
    atomic_json(out / 'stats.json', {'samples': n_samples, 'nonpad_tokens': sum(len(r['input_ids']) for r in train),
        'supervised_tokens': sum(sum(r['loss_mask']) for r in train), 'length_histogram': hist,
        'source': source, 'revision': revision, 'seq_length': SEQ_LENGTH, 'max_tokens': MAX_TOKENS,
        'system': SYSTEM, 'canonical_sha256': file_hash(out / 'canonical.jsonl'),
        'mapping_sha256': digest(selected), 'preprocessor_sha256': file_hash(repo / 'eagle/traineagle3/main.py')})
    print(f'Saved {n_samples} identical author-tokenized examples and shared 32K mapping')
    # Streaming `datasets` leaves background threads that crash CPython's finalization
    # ("PyGILState_Release ... must be current"), turning a finished run into exit code 134.
    # Every output is written and fsynced above, so leave without finalization.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main(*sys.argv[1:])
