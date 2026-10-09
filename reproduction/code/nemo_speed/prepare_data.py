"""Task 4 data: one tokenized corpus and one 32K draft vocabulary, shared by both trainers.

    python prepare_data.py <model dir> <N examples> <out dir> <EAGLE repo>
    env: TRAIN_DATA_REVISION (pinned revision of the dataset, set by orchestrate.py)

Source: frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct (answers already written by the target).
Each dialogue is tokenized by the authors' own preprocess_function (taken from traineagle3/main.py and run as is),
so the token ids and loss masks are exactly what the authors' trainer would see; we also check that the plain
Llama-3 chat template gives the same ids. Examples longer than 1900 tokens are skipped.

Writes to <out dir>:
    canonical.jsonl         N training examples: input_ids, attention_mask, loss_mask
    test_canonical.jsonl    16 more (the authors' trainer wants a test set)
    sharegpt.jsonl / messages.jsonl   the same dialogues in the two source formats, for reference
    selected_tokens.json    the 32K most frequent supervised tokens (the draft vocabulary)
    stats.json              counts, length histogram and hashes of the above
"""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent.parent / 'inference_speed'
sys.path.insert(0, str(PACKAGE))
from common import SYSTEM, atomic_json, atomic_text, digest, file_hash   # noqa: E402

SOURCE = 'frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct'
MAX_TOKENS = 1900
SEQ_LENGTH = 2048
TEST_EXAMPLES = 16


def author_preprocessor(main_py, tok):
    """Pull build_dataset_rank.preprocess_function out of the authors' main.py and compile only that function."""
    import torch
    tree = ast.parse(Path(main_py).read_text(encoding='utf-8'))
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dataset_rank')
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_function')
    namespace = {'torch': torch, 'tokenizer': tok, 'train_config': {'max_len': SEQ_LENGTH}}
    module = ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[]))
    exec(compile(module, str(main_py), 'exec'), namespace)
    return namespace['preprocess_function']


def selected_vocab(rows, vocab_size, size=32000):
    """The `size` most frequent supervised tokens (ties by id), topped up with the lowest unused ids; sorted."""
    counts = Counter(t for r in rows for t, m in zip(r['input_ids'], r['loss_mask']) if m)
    selected = [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:size]]
    used = set(selected)
    selected.extend(i for i in range(vocab_size) if i not in used)
    selected = sorted(selected[:size])
    if len(selected) != size or min(selected) < 0 or max(selected) >= vocab_size:
        raise ValueError('Invalid shared draft vocabulary')
    return selected


def usable(dialogue):
    if not isinstance(dialogue, list) or len(dialogue) < 2 or len(dialogue) % 2:
        return False
    if [m['role'] for m in dialogue] != ['user', 'assistant'] * (len(dialogue) // 2):
        return False
    return all(isinstance(m.get('content'), str) and m['content'].strip() for m in dialogue)


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
    revision = os.environ['TRAIN_DATA_REVISION']

    rows, originals, messages = [], [], []
    for item in load_dataset(SOURCE, revision=revision, split='train', streaming=True):
        dialogue = item['conversations']
        if not usable(dialogue):
            continue
        rid = str(item['id'])
        sharegpt = [{'from': 'human' if m['role'] == 'user' else 'gpt', 'value': m['content']} for m in dialogue]
        pre = preprocess({'id': [rid], 'conversations': [sharegpt]})
        if not pre['input_ids']:
            continue
        ids = pre['input_ids'][0].tolist()[0]
        mask = pre['loss_mask'][0].tolist()[0]
        if not 0 < len(ids) <= MAX_TOKENS or not sum(mask):
            continue
        chat = [{'role': 'system', 'content': SYSTEM}] + dialogue
        if tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False) != ids:
            raise ValueError('Chat template parity failure')
        rows.append({'id': rid, 'input_ids': ids, 'attention_mask': [1] * len(ids), 'loss_mask': mask})
        originals.append({'id': rid, 'conversations': sharegpt})
        messages.append({'id': rid, 'messages': chat})
        if len(rows) == n_samples + TEST_EXAMPLES:
            break
    if len(rows) != n_samples + TEST_EXAMPLES:
        raise ValueError(f'Not enough valid examples: {len(rows)}')

    train, test = rows[:n_samples], rows[n_samples:]
    for name, data in [('canonical.jsonl', train), ('test_canonical.jsonl', test),
                       ('sharegpt.jsonl', originals[:n_samples]), ('messages.jsonl', messages[:n_samples])]:
        atomic_text(out / name, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    vocab_size = json.loads((Path(model_dir) / 'config.json').read_text(encoding='utf-8'))['vocab_size']
    selected = selected_vocab(train, vocab_size)
    atomic_json(out / 'selected_tokens.json', selected)
    histogram = {'0-512': 0, '513-1024': 0, '1025-1900': 0}
    for r in train:
        n = len(r['input_ids'])
        histogram['0-512' if n <= 512 else ('513-1024' if n <= 1024 else '1025-1900')] += 1
    atomic_json(out / 'stats.json', {
        'samples': n_samples, 'nonpad_tokens': sum(len(r['input_ids']) for r in train),
        'supervised_tokens': sum(sum(r['loss_mask']) for r in train), 'length_histogram': histogram,
        'source': SOURCE, 'revision': revision, 'seq_length': SEQ_LENGTH, 'max_tokens': MAX_TOKENS,
        'system': SYSTEM, 'canonical_sha256': file_hash(out / 'canonical.jsonl'),
        'mapping_sha256': digest(selected), 'preprocessor_sha256': file_hash(repo / 'eagle/traineagle3/main.py')})
    print(f'Saved {n_samples} identical author-tokenized examples and shared 32K mapping')
    # Streaming `datasets` leaves background threads that crash CPython's shutdown ("PyGILState_Release"),
    # turning a finished run into a failure. Everything is written above, so skip the shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main(*sys.argv[1:])
