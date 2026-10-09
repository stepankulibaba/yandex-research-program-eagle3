"""Shared by both training-speed runs (the authors' trainer and NeMo): the same data, vocabulary and stopwatch.

- load_rows / original_dataset / nemo_loader: one tokenized corpus (data/canonical.jsonl, built by
  ../nemo_speed/prepare_data.py with the authors' own preprocessing), fed to both trainers.
- mapping: one 32K draft vocabulary for both.
- SpeedWindow: measures time and tokens over optimizer steps after a warm-up.
"""
import json
from pathlib import Path
import time

from common import SCHEMA, TRAINING_PROTOCOL, read_json

ROW = 2048            # every training row is 2048 tokens (padded or packed)
MAX_EXAMPLE = 1900    # longest example kept by prepare_data.py


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for r in rows:
        n = len(r['input_ids'])
        if not 0 < n <= MAX_EXAMPLE or len(r['loss_mask']) != n or r['attention_mask'] != [1] * n:
            raise ValueError('Invalid canonical training row')
    return rows


def original_dataset(path):
    """The corpus in the format of the authors' trainer: one example per item, tensors of shape [1, n]."""
    from datasets import Dataset
    rows = load_rows(path)
    data = Dataset.from_list([{k: [r[k]] for k in ('input_ids', 'attention_mask', 'loss_mask')} for r in rows])
    data.set_format(type='torch')
    return data


def nemo_loader(path, *, packed, batch_size, pad_id=0):
    """The corpus as a NeMo EAGLE-3 dataloader, built with NeMo's own packing and collate functions.

    packed=False: one example per 2048-token row, padded (the authors' trainer gets the same padding).
    packed=True:  examples concatenated into 2048-token rows (NeMo packing).
    Order is fixed (no shuffle), the same as in the authors' adapted trainer.
    """
    from torch.utils.data import DataLoader
    from nemo_automodel.components.datasets.llm.eagle3 import (_pack_collate, _stack_batch,
                                                                build_packed_eagle3_dataset)
    rows = load_rows(path)
    if packed:
        data = build_packed_eagle3_dataset(rows, packed_sequence_size=ROW, pad_token_id=pad_id)
        for row in data:
            lengths = list(row['seq_lens'])                 # NeMo folds the trailing pad into the last document
            lengths[-1] -= ROW - sum(row['attention_mask'])
            row['_document_lengths'] = lengths
        collate_rows = _pack_collate
    else:
        data = []
        for row in rows:
            n = len(row['input_ids'])
            data.append({'input_ids': row['input_ids'] + [pad_id] * (ROW - n),
                         'attention_mask': [1] * n + [0] * (ROW - n),
                         'loss_mask': row['loss_mask'] + [0] * (ROW - n),
                         '_document_lengths': [n]})
        collate_rows = _stack_batch

    def collate(features):
        batch = collate_rows(features)
        batch['_document_lengths'] = [n for f in features for n in f['_document_lengths']]
        return batch
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)


def mapping(path, vocab_size):
    """The shared 32K draft vocabulary: (target ids of the draft tokens, boolean mask over the target vocab)."""
    import torch
    ids = torch.tensor(read_json(path), dtype=torch.long)
    if ids.numel() != 32000 or ids.unique().numel() != 32000 or ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError('Bad draft-vocab mapping')
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[ids] = True
    return ids, mask


class SpeedWindow:
    """Stopwatch over optimizer steps `warmup + 1 .. expected_steps`.

    Each trainer calls before_batch() for every micro-batch (still on CPU, so counting costs no GPU sync) and
    after_step() after every optimizer step. The GPU is synchronised only at the start and the end of the window.
    """

    def __init__(self, warmup, expected_steps, framework, config):
        if warmup < 0 or expected_steps <= warmup:
            raise ValueError('Not enough optimizer steps after warm-up')
        self.warmup, self.expected_steps = warmup, expected_steps
        self.framework, self.config = framework, config
        self.start = self.end = None
        self.steps = self.nonpad = self.padded = self.supervised = self.documents = self.rows = 0
        self.finite_loss = True
        self.length_histogram = {'0-512': 0, '513-1024': 0, '1025-1900': 0}

    def before_batch(self, optimizer_step, batch, torch):
        if optimizer_step < self.warmup:
            return
        if self.start is None:                      # first measured micro-batch: start the clock
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()
        self.nonpad += int(batch['attention_mask'].sum())
        self.padded += batch['input_ids'].numel()
        self.supervised += int(batch['loss_mask'].sum())
        self.rows += batch['input_ids'].shape[0]
        lengths = batch.get('_document_lengths')
        if lengths is None:
            lengths = batch['attention_mask'].sum(dim=1).tolist()
        for length in lengths:
            self.documents += 1
            bucket = '0-512' if length <= 512 else ('513-1024' if length <= 1024 else '1025-1900')
            self.length_histogram[bucket] += 1

    def after_step(self, completed_steps, torch):
        if completed_steps > self.warmup:
            self.steps += 1
        if completed_steps == self.expected_steps:  # last step: stop the clock
            torch.cuda.synchronize()
            self.end = time.perf_counter()

    def result(self, torch):
        if self.start is None or self.end is None or not self.finite_loss:
            raise ValueError('Incomplete timing window or nonfinite loss')
        return {'schema': SCHEMA, 'protocol': TRAINING_PROTOCOL, 'framework': self.framework,
                'complete': True, 'finite_loss': True, 'seconds': self.end - self.start,
                'optimizer_steps': self.steps, 'warmup_optimizer_steps': self.warmup,
                'nonpad_tokens': self.nonpad, 'padded_tokens': self.padded, 'supervised_tokens': self.supervised,
                'documents': self.documents, 'rows': self.rows, 'length_histogram': self.length_histogram,
                'peak_allocated_gb': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30, 'config': self.config}
