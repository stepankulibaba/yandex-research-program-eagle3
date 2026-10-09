"""Canonical CPU loaders used by both adapters. No implicit chat reformatting."""
import json
from pathlib import Path
from runtime import read_json


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for r in rows:
        n = len(r['input_ids'])
        if not 0 < n <= 1900 or len(r['loss_mask']) != n or r['attention_mask'] != [1] * n:
            raise ValueError('Invalid canonical training row')
    return rows


def original_dataset(path):
    import torch
    from datasets import Dataset
    rows = load_rows(path)
    result = Dataset.from_list([{k: [r[k]] for k in ('input_ids', 'attention_mask', 'loss_mask')} for r in rows])
    result.set_format(type='torch')
    return result


def canonical_nemo_loader(path, *, packed, batch_size, pad_id=0):
    import torch
    from torch.utils.data import DataLoader
    from nemo_automodel.components.datasets.llm.eagle3 import build_packed_eagle3_dataset, _pack_collate, _stack_batch
    rows = load_rows(path)
    # A fixed deterministic order is shared by both protocols; shuffle on load
    # is intentionally replaced by a fixed order in the original adapter too.
    if packed:
        data = build_packed_eagle3_dataset(rows, packed_sequence_size=2048, pad_token_id=pad_id)
        for r in data:
            lengths = list(r['seq_lens'])
            lengths[-1] -= 2048 - sum(r['attention_mask'])
            r['_document_lengths'] = lengths
        collator = _pack_collate
    else:
        data = []
        for row in rows:
            n = len(row['input_ids'])
            data.append({'input_ids': row['input_ids'] + [pad_id] * (2048 - n),
                         'attention_mask': [1] * n + [0] * (2048 - n),
                         'loss_mask': row['loss_mask'] + [0] * (2048 - n),
                         '_document_lengths': [n]})
        collator = _stack_batch

    def collate(features):
        batch = collator(features)
        batch['_document_lengths'] = [n for f in features for n in f['_document_lengths']]
        return batch
    # CPU collation before timing keeps counts exact without GPU .item syncs.
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)


def mapping(path, vocab_size):
    import torch
    ids = torch.tensor(read_json(path), dtype=torch.long)
    if ids.numel() != 32000 or ids.unique().numel() != 32000 or ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError('Bad draft-vocab mapping')
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[ids] = True
    return ids, mask
