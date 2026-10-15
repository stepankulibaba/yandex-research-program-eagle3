"""Variant `ours`: NeMo AutoModel's EAGLE-3 recipe with our layers and fc_norm; resumable across 8-hour sessions.

    <NeMo venv>/bin/python train_nemo_full.py [--deadline UNIX_TIME] [--runs DIR] [--data DIR] [--smoke]
    (session.py starts it with the working directory S.NEMO_DIR)

The recipe (nemo_automodel.recipes.llm.train_eagle3, pinned commit) runs with its own LLaMA-3.1-8B / PerfectBlend
settings (settings.NEMO_RECIPE): AdamW (0.9, 0.95) lr 2e-4, 5 % warmup and cosine to 0.1x, clip 1.0, 4 TTT steps,
32 packed rows of 2048 tokens per optimizer step, bf16 compute; plus the speed-ups of the speed study: packing and
torch.compile of the draft. Ours: aux_layer_ids [15, 21, 28] (= inputs of layers 16, 22, 29) and fc_norm.

What is replaced before the recipe starts (nothing else of it changes):
0. precision: the recipe casts the draft to bf16 and runs AdamW on bf16 weights, so every update smaller than
   half a bf16 step is rounded away (all RMSNorm scales stayed exactly 1.0, and most weight updates are lost once
   the lr is below ~1e-4). Here the draft weights and the AdamW state are fp32 and the forward runs under bf16
   autocast: mixed precision with fp32 master weights, as the baseline's DeepSpeed fp16 has;
1. data: the baseline's tokenized dialogues (data/train) in the baseline's epoch-0 order, packed greedily into
   2048-token rows; the recipe restores only the step number on resume, so the sampler skips the rows already trained;
2. the draft vocabulary: data/vocab.json (the baseline's);
3. after each optimizer step: metrics / status for monitor.py, bot.py and clearml_sync.py (the same files as
   train.py), validation on the baseline's 400 held-out dialogues, checkpoints every CHECKPOINT_MINUTES, before the
   session deadline, on SIGTERM / SIGUSR1 / runs/ours/STOP, and exported draft weights;
4. a skipped optimizer step when the gradient norm is non-finite or > 10 (as DeepSpeed's fp16 overflow skip);
5. a divergence guard (settings.NEMO_GUARD): a loss jump, a huge or non-finite grad norm -> the checkpoints of the
   last steps are dropped, the lr is halved (runs/ours/guard.json) and the process exits with code 3, so that
   session.py / start_ours_now.sh start it again from the last healthy checkpoint.

Validation is the authors' measure, so the curves of both variants are comparable: per-dialogue accuracy of each of
7 draft steps (the draft trains 4; steps 5-7 show how it extrapolates, as it must at inference depth 7) and
tau_estimate = 1 + sum of products of the accuracies. One difference stays: NeMo shifts the loss mask by one token
(it supervises the prediction of token t+2 when token t+1 is the answer's), the authors do not.
"""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import settings as S
from monitor import jsonl as monitor_jsonl
from train import atomic_json, epoch_order, log, tau_estimate

VAL_STEPS = 7                 # draft steps scored at validation (the authors' TTT length)


class StopRun(Exception):
    """Raised after the stop checkpoint: leaves the recipe's loop through its own cleanup."""


class Rollback(Exception):
    """Training diverged: the newest checkpoints are dropped, the lr is lowered; exit code 3 = start me again."""


def keep_until(path, step):
    """Drop the lines of a .jsonl log written after `step` (they belong to a run state that was rolled back)."""
    if path.exists():
        lines = [x for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        kept = []
        for x in lines:
            try:
                if json.loads(x)['step'] <= step:
                    kept.append(x)
            except ValueError:
                pass
        path.write_text(''.join(x + '\n' for x in kept), encoding='utf-8')


def pack_rows(lengths, order, row):
    """Greedy packing of dialogues (in `order`) into rows of at most `row` tokens -> (flat indices, row offsets)."""
    flat, offsets, used = [], [0], 0
    for i in order:
        n = min(int(lengths[i]), row)
        if used and used + n > row:
            offsets.append(len(flat))
            used = 0
        flat.append(int(i))
        used += n
    offsets.append(len(flat))
    return flat, offsets


def dialogue_lengths(dataset):
    import numpy as np
    import pyarrow.compute as pc
    if getattr(dataset, '_indices', None) is not None:
        raise ValueError('expected a dataset saved without an index mapping')
    return np.asarray(pc.list_value_length(dataset.data.column('input_ids')).to_numpy(zero_copy_only=False))


class PackedRows:
    """Rows of NeMo's packed format (build_packed_eagle3_dataset on the dialogues of one row)."""

    def __init__(self, dataset, flat, offsets, row, pad_id):
        self.dataset, self.flat, self.offsets, self.row, self.pad_id = dataset, flat, offsets, row, pad_id

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, k):
        from nemo_automodel.components.datasets.llm.eagle3 import build_packed_eagle3_dataset
        docs = [self.dataset[i] for i in self.flat[self.offsets[k]:self.offsets[k + 1]]]
        packed = build_packed_eagle3_dataset(docs, packed_sequence_size=self.row, pad_token_id=self.pad_id)
        if len(packed) != 1:
            raise RuntimeError(f'row {k} did not pack into one row')
        return packed[0]


class SkipSampler:
    """All row indices in order, starting after the rows the restored optimizer steps already consumed."""

    def __init__(self, n, first):
        self.n, self.first = n, first           # first(): evaluated when the epoch's iteration starts

    def __len__(self):
        return self.n                           # full length: the lr schedule is computed from it

    def __iter__(self):
        return iter(range(min(self.first(), self.n), self.n))


def diagnose(trainer, ckpts, collate, torch, row_starts, n_rows=16):
    """For each checkpoint: loss and gradient norm per draft weight, on the same packed rows (no optimizer step)."""
    starts = [int(x) for x in row_starts.split(',')]
    rows = [r for s in starts for r in range(s, s + n_rows)]
    draft = trainer._module().draft_model
    names = [n for n, p in draft.named_parameters() if p.requires_grad]
    trainer.new_window()
    trainer.trainer_module.train()
    for ckpt in ckpts:
        trainer.checkpointer.load_model(draft, str(Path(ckpt) / 'model'))
        per = {n: [] for n in names}
        totals, losses = [], []
        for r in rows:
            draft.zero_grad(set_to_none=True)
            metrics = trainer._forward_batch(collate([trainer.train_dataloader.dataset[r]]))
            metrics.loss.backward()
            sq = 0.0
            for n, p in draft.named_parameters():
                if p.grad is not None:
                    g = float(p.grad.float().norm())
                    per[n].append(g)
                    sq += g * g
            totals.append(sq ** 0.5)
            losses.append(float(metrics.loss))
        log(f'== {Path(ckpt).name}')
        for s in starts:
            idx = [i for i, r in enumerate(rows) if s <= r < s + n_rows]
            t = sorted(totals[i] for i in idx)
            log(f'rows {s}..{s + n_rows - 1}: loss {sum(losses[i] for i in idx) / len(idx):.3f}, grad norm median '
                f'{t[len(t) // 2]:.3g} max {t[-1]:.3g}')
        ranked = sorted(names, key=lambda n: -sum(per[n]) / max(1, len(per[n])))
        for n in ranked[:12]:
            v = sorted(per[n])
            log(f'   {n:52s} grad median {v[len(v) // 2]:.3g} max {v[-1]:.3g}   |w| {float(dict(draft.named_parameters())[n].norm()):.4g}')
    draft.zero_grad(set_to_none=True)



def recipe_config(args, run_dir, accum):
    spec, R = S.VARIANTS['ours'], S.NEMO_RECIPE
    return {
        'recipe': 'TrainEagle3Recipe',
        'dist_env': {'backend': 'nccl', 'timeout_minutes': 60},
        'recipe_args': {
            'target_model_name_or_path': str(S.TARGET),
            'train_data_path': str(Path(args.data) / 'train'),     # read by our dataloader, not by the recipe's
            'val_data_path': None,                                 # validation is ours (the authors' measure)
            'output_dir': str(run_dir / 'nemo'),
            'seq_length': R['packed_sequence_size'],
            'packed_sequence_size': R['packed_sequence_size'],
            'micro_batch_size': R['micro_batch_size'],
            'grad_accumulation_steps': accum,
            'num_workers': 4,
            'num_epochs': 1,
            'ttt_steps': R['ttt_steps'],
            'draft_vocab_size': R['draft_vocab_size'],
            'freeze_embeddings': R['freeze_embeddings'],
            'trust_remote_code': False,
            'shuffle_seed': S.SEED,
            'log_every_steps': args.log_every,
            'max_grad_norm': R['max_grad_norm'],
            'aux_layer_ids': [layer - 1 for layer in spec['layers']],   # NeMo counts layer outputs
            'fc_norm': spec['fc_norm'],
            'target_force_hf': True,                               # the HF LLaMA, as in the speed study
            'target_attn_implementation': 'sdpa',                  # block-causal 4D mask for the packed rows
            'draft_attn_implementation': 'eager',
        },
        'optimizer': {k: R[k] for k in ('lr', 'betas', 'weight_decay', 'warmup_ratio', 'min_lr_ratio')},
        'checkpoint': {'enabled': True, 'checkpoint_dir': str(run_dir / 'checkpoints'),
                       'model_save_format': 'safetensors', 'save_consolidated': 'final',
                       'max_recent_checkpoints': S.NEMO_KEEP_CHECKPOINTS},
        **({'compile': {'enabled': True, 'mode': 'default'}} if S.NEMO_COMPILE else {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, default=0, help='unix time; stop with a checkpoint before it')
    parser.add_argument('--margin-minutes', type=float, default=12)
    parser.add_argument('--data', default=str(S.DATA))
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--log-every', type=int, default=10, help='optimizer steps (~1000 dialogues each 10)')
    parser.add_argument('--val-every', type=int, default=S.VAL_EVERY * S.GLOBAL_BATCH, help='dialogues')
    parser.add_argument('--val-eval', type=int, default=S.VAL_EVAL)
    parser.add_argument('--checkpoint-minutes', type=float, default=S.NEMO_CHECKPOINT_MINUTES)
    parser.add_argument('--milestone-every', type=int, default=S.MILESTONE_EVERY * S.GLOBAL_BATCH, help='dialogues')
    parser.add_argument('--smoke', action='store_true', help='a few hundred dialogues, accumulation 4')
    parser.add_argument('--diagnose', default='', help='comma-separated checkpoint dirs: grad norms per weight')
    parser.add_argument('--diagnose-rows', default='22000,27488', help='first rows of the two 16-row samples')
    args = parser.parse_args()
    data, runs = Path(args.data), Path(args.runs)
    run_dir = runs / 'ours'
    for d in (run_dir, run_dir / 'milestones'):
        d.mkdir(parents=True, exist_ok=True)
    accum = 4 if args.smoke else S.NEMO_RECIPE['grad_accumulation_steps']
    deadline = args.deadline or float('inf')
    plan = json.loads((runs / 'plan.json').read_text(encoding='utf-8')) if (runs / 'plan.json').exists() else {}
    compare_samples = plan.get('compare_step', 0) * S.GLOBAL_BATCH or None   # where baseline stops (dialogues)
    if args.smoke:          # 600 dialogues: every code path (validation, milestones, checkpoints, resume) in minutes
        args.log_every, args.val_every, args.val_eval, args.milestone_every = 2, 200, 40, 400
        args.checkpoint_minutes, compare_samples = 0.2, 300
    G = dict(S.NEMO_GUARD, margin_steps=4, from_step=10) if args.smoke else S.NEMO_GUARD
    force_bad = int(os.environ.get('BT_FORCE_DIVERGE_AT', 0))       # smoke test of the rollback path
    force_skip = int(os.environ.get('BT_FORCE_SKIP_AT', 0))         # smoke test of the skipped step
    guard_path = run_dir / 'guard.json'           # lr scale and rollbacks: survive the rollback itself
    guard = json.loads(guard_path.read_text(encoding='utf-8')) if guard_path.exists() else {'scale': 1.0, 'rollbacks': []}

    import torch
    from datasets import load_from_disk
    import nemo_automodel.recipes.llm.train_eagle3 as recipe
    from nemo_automodel.components.datasets.llm.eagle3 import _pack_collate
    from nemo_automodel.components.speculative.eagle import core as eagle_core

    stop = {'signal': None}
    faulthandler.register(signal.SIGUSR2, all_threads=True)          # kill -USR2 <pid>: stacks to train.log
    for name in ('SIGTERM', 'SIGUSR1'):
        signal.signal(getattr(signal, name), lambda s, f: stop.update(signal=signal.Signals(s).name))

    train = load_from_disk(str(data / 'train'))
    val = load_from_disk(str(data / 'val'))
    order = epoch_order(len(train), 0)                                 # the baseline's order of epoch 0
    if args.smoke:
        order = order[:600]
    row = S.NEMO_RECIPE['packed_sequence_size']
    flat, offsets = pack_rows(dialogue_lengths(train), order, row)
    n_val = min(args.val_eval, len(val))
    val_flat, val_offsets = pack_rows(dialogue_lengths(val), range(n_val), row)
    selected = torch.tensor(json.loads((data / 'vocab.json').read_text(encoding='utf-8')), dtype=torch.long)
    data_stats = json.loads((data / 'stats.json').read_text(encoding='utf-8'))
    epoch_dialogues = len(order)

    state_path = run_dir / 'nemo_state.json'      # our counters, saved next to each NeMo checkpoint
    state = {'samples': 0, 'tokens': 0, 'train_seconds': 0.0, 'val_history': [], 'best_val_tau': None,
             'milestones': [], 'rows_per_step': accum, 'train_ids_sha256': data_stats['train_ids_sha256'],
             'layers': S.VARIANTS['ours']['layers'], 'fc_norm': S.VARIANTS['ours']['fc_norm'], 'smoke': args.smoke,
             'master_weights': 'fp32'}

    class Ours(recipe.TrainEagle3Recipe):
        def _setup_online_target(self, *a, **kw):
            out = super()._setup_online_target(*a, **kw)     # the frozen target: bf16, as in the recipe
            self.compute_dtype = torch.float32               # the draft is built next: fp32 master weights
            return out

        def _build_train_dataloader(self, data_path, split):
            pad = self.tokenizer.pad_token_id
            rows = PackedRows(train, flat, offsets, row, pad if pad is not None else self.tokenizer.eos_token_id)
            sampler = SkipSampler(len(rows), lambda: self.runtime.global_step * self.grad_accumulation_steps)
            return torch.utils.data.DataLoader(rows, batch_size=1, sampler=sampler, collate_fn=_pack_collate,
                                               num_workers=4, pin_memory=True, persistent_workers=False)

        def _forward_batch(self, batch, target_batch=None):
            with torch.autocast('cuda', dtype=torch.bfloat16):    # the backward runs outside (in the recipe)
                metrics = super()._forward_batch(batch, target_batch)
            if self.trainer_module.training:
                w = self.window
                w['rows'] += 1
                w['docs'] += batch['seq_lens'].gt(0).sum()            # CPU tensors: no GPU sync
                w['tokens'] += batch['attention_mask'].sum()
                w['hit1'] += metrics.step_prefix_hits[0]               # step 1: hits / all answer positions,
                w['valid1'] += metrics.step_valid[0]                   # the authors' accuracy of step 1
                w['loss'] += metrics.loss.detach()
            return metrics

        def _make_progress_bar(self, *a, **kw):
            return None                                                # train.log gets our lines instead

        def _wandb_log(self, data, step):
            self.last_log = data                                       # lr, grad_norm, NeMo's window metrics

        def _maybe_save_step_checkpoint(self, epoch):
            gs = self.runtime.global_step
            w = self.window
            if gs % self.log_every_steps == 0 or gs >= self.total_optim_steps:
                self.write_metrics(epoch)
            if state['samples'] >= self.next_val or gs >= self.total_optim_steps:
                self.validate()
                self.next_val = (state['samples'] // args.val_every + 1) * args.val_every
            if state['samples'] >= self.next_milestone:
                self.export(f'd{self.next_milestone:08d}')
                self.next_milestone += args.milestone_every
            if compare_samples and state['samples'] >= compare_samples and 'compare' not in state['milestones']:
                if state['val_history'][-1]['step'] != gs:
                    self.validate()                # where baseline stops: both are compared here
                self.export('compare')
            if gs >= self.total_optim_steps:      # the recipe saves the final checkpoint itself, right after:
                state['step'] = gs                 # our counters for it go first
                atomic_json(state_path, state)
                return False
            reason, finish = None, False
            if stop['signal']:
                reason, finish = f'signal {stop["signal"]}', True
            elif (run_dir / 'STOP').exists():
                reason, finish = 'STOP file', True
            elif time.time() > deadline - args.margin_minutes * 60:
                reason, finish = 'session deadline', True
            elif (run_dir / 'CKPT').exists():
                reason = 'CKPT file'
                (run_dir / 'CKPT').unlink()
            elif time.time() - self.last_checkpoint > args.checkpoint_minutes * 60:
                reason = 'periodic'
            if reason:
                self.save(epoch, reason)
            if finish:
                raise StopRun(reason)
            return bool(reason)

        def save(self, epoch, reason):
            t0 = time.time()
            self.flush_window()
            state['step'] = self.runtime.global_step
            atomic_json(state_path, state)
            self.save_checkpoint(epoch=epoch, step=self.runtime.global_step, best_metric_key='val_loss')
            self.checkpointer.async_wait()
            self.checkpointer.lifecycle.complete_pending()
            path = Path(self.checkpoint_config.checkpoint_dir) / f'epoch_{epoch}_step_{self.runtime.global_step}'
            shutil.copy(state_path, path / 'nemo_state.json')
            self.last_checkpoint = time.time()
            atomic_json(run_dir / 'latest_ok.json', {'tag': path.name, 'step': self.runtime.global_step,
                                                     'samples': state['samples'], 'time': time.time(),
                                                     'reason': reason})
            log(f'checkpoint {path.name} ({reason}) in {time.time() - t0:.0f} s')

        def rollback(self, why):
            gs = self.runtime.global_step
            root = Path(self.checkpoint_config.checkpoint_dir)
            if len(guard['rollbacks']) >= G['max_rollbacks']:
                self.write_status('stopped')
                log(f'DIVERGED at step {gs} ({why}); {len(guard["rollbacks"])} rollbacks already: stopping for a human')
                raise StopRun('diverged, rollbacks exhausted')
            for d in root.glob('epoch_*_step_*'):            # checkpoints that may already hold the spike
                if int(d.name.rsplit('_', 1)[1]) > gs - G['margin_steps']:
                    shutil.rmtree(d, ignore_errors=True)
            for name in ('LATEST', 'LATEST.txt'):            # NeMo then resumes from the newest remaining one
                if os.path.lexists(root / name):
                    os.unlink(root / name)
            guard['rollbacks'].append({'step': gs, 'samples': state['samples'], 'why': why, 'time': time.time(),
                                       'scale_before': guard['scale']})
            guard['scale'] *= G['lr_factor']
            atomic_json(guard_path, guard)
            self.write_status('rollback')
            log(f'DIVERGED at step {gs} ({why}): checkpoints after step {gs - G["margin_steps"]} dropped, '
                f'lr scale now {guard["scale"]}; restarting from the last healthy checkpoint')
            raise Rollback(why)

        def new_window(self):
            z = lambda: torch.zeros((), device=self.device)
            self.window = {'start': time.time(), 'rows': 0, 'docs': torch.zeros((), dtype=torch.long),
                           'tokens': torch.zeros((), dtype=torch.long), 'hit1': z(), 'valid1': z(), 'loss': z()}

        def flush_window(self):
            """Counters of the window into the state (before a checkpoint or a metric line)."""
            w = self.window
            docs, tokens = int(w['docs']), int(w['tokens'])
            state['samples'] += docs
            state['tokens'] += tokens
            elapsed = time.time() - w['start']
            state['train_seconds'] += elapsed
            out = {'rows': w['rows'], 'docs': docs, 'tokens': tokens, 'seconds': elapsed,
                   'loss': float(w['loss']) / max(1, w['rows']),
                   'acc1': float(w['hit1']) / max(1.0, float(w['valid1']))}
            self.new_window()
            return out

        def write_metrics(self, epoch):
            w = self.flush_window()
            if not w['rows']:
                return
            nonfinite = not math.isfinite(w['loss'])
            state['nonfinite'] = state.get('nonfinite', 0) + int(nonfinite)
            info = getattr(self, 'last_log', {}) or {}
            gn = info.get('train/grad_norm')          # NeMo logs it after this hook: the previous window's
            why = None
            if nonfinite:
                why = 'loss is not finite'
            elif gn is not None and not (math.isfinite(gn) and gn < G['grad_norm']):
                why = f'grad norm {gn:.3g}'
            elif self.runtime.global_step >= G['from_step'] and w['loss'] > self.best_loss + G['loss_jump']:
                why = f'loss {w["loss"]:.2f} > best {self.best_loss:.2f} + {G["loss_jump"]}'
            elif force_bad and self.runtime.global_step >= force_bad and not guard['rollbacks']:
                why = 'forced (smoke test)'
            if why:
                self.rollback(why)
            self.best_loss = min(self.best_loss, w['loss'])
            line = {'step': self.runtime.global_step, 'samples': state['samples'], 'epoch': epoch,
                    'tokens_per_s': w['tokens'] / w['seconds'], 'samples_per_s': w['docs'] / w['seconds'],
                    'rows_per_s': w['rows'] / w['seconds'], 'lr': self.lr_scheduler.get_last_lr()[0],
                    'grad_norm': info.get('train/grad_norm'), 'loss_scale': None,
                    'skipped_steps': state.get('skipped_steps', 0),
                    'nonfinite': state['nonfinite'],
                    'ploss': [w['loss']], 'acc': [w['acc1']],                # NeMo's loss (4-step mean); step-1 acc
                    'tau_sim_train': info.get('train/tau_sim'), 'time': time.time()}
            with open(run_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(line) + '\n')
            self.write_status('running', line)

        def write_status(self, phase, window=None):
            latest = run_dir / 'latest_ok.json'
            atomic_json(run_dir / 'status.json', {
                'variant': 'ours', 'trainer': 'nemo', 'phase': phase, 'pid': os.getpid(), 'updated': time.time(),
                'step': self.runtime.global_step, 'total_steps': self.total_optim_steps,
                'stop_step': self.total_optim_steps, 'epoch': 0, 'rows_per_step': accum,
                'samples': state['samples'], 'stop_samples': epoch_dialogues, 'train_dialogues': epoch_dialogues,
                'tokens': state['tokens'], 'nonfinite': state.get('nonfinite', 0),
                'skipped_steps': state.get('skipped_steps', 0),
                'lr_scale': guard['scale'], 'rollbacks': guard['rollbacks'],
                'lr': self.lr_scheduler.get_last_lr()[0], 'loss_scale': None, 'window': window,
                'last_val': state['val_history'][-1] if state['val_history'] else None,
                'best_val_tau': state['best_val_tau'],
                'last_checkpoint': json.loads(latest.read_text()) if latest.exists() else None,
                'deadline': None if deadline == float('inf') else deadline, 'train_seconds': state['train_seconds'],
                'gpu_memory_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                'disk_free_gb': shutil.disk_usage(run_dir).free / 2 ** 30})

        @torch.no_grad()
        @torch.autocast('cuda', dtype=torch.bfloat16)
        def validate(self):
            """The authors' validation measure on the baseline's held-out dialogues (see the module docstring)."""
            t0 = time.time()
            module = self._module()
            draft = module.draft_model
            was_training = self.trainer_module.training
            self.trainer_module.eval()
            pad = self.tokenizer.pad_token_id
            rows = PackedRows(val, val_flat, val_offsets, row, pad if pad is not None else self.tokenizer.eos_token_id)
            hits, dens, ce, lengths = [], [], [], []
            chain_hits = torch.zeros(VAL_STEPS, device=self.device)
            chain_valid = torch.zeros(VAL_STEPS, device=self.device)
            for k in range(len(rows)):
                batch = {key: v.to(self.device) for key, v in _pack_collate([rows[k]]).items()}
                tb = self.target_wrapper.generate_batch(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    loss_mask=batch['loss_mask'], position_ids=batch['position_ids'], seq_lens=batch['seq_lens'],
                    doc_remaining=batch['doc_remaining']).to_trainer_inputs()
                target_probs, position_mask = eagle_core._compute_target_distribution(
                    tb['target_logits'], module.selected_token_ids, module.selected_token_mask, tb['loss_mask'])
                hidden = draft.project_hidden_states(tb['aux_hidden_states'])
                doc = (batch['position_ids'][0] == 0).long().cumsum(0) - 1           # document of each slot
                n_docs = int(doc.max()) + 1
                real = batch['attention_mask'][0].float()
                lengths.append(torch.zeros(n_docs, device=self.device).index_add_(0, doc, real))
                ids, pmask, probs, lmask = tb['input_ids'], position_mask, target_probs, tb['loss_mask'].bool()
                cache = [[], []]
                h_row, d_row, c_row = [], [], []
                prefix_ok = prefix_valid = None
                for step in range(VAL_STEPS):
                    hidden = draft(input_ids=ids, projected_hidden_states=hidden,
                                   attention_mask=batch['attention_mask'], position_ids=batch['position_ids'],
                                   cache_hidden=cache, seq_lens=batch['seq_lens'])
                    logits = draft.compute_logits(hidden).float()
                    in_doc = step < batch['doc_remaining']
                    sup = (pmask.squeeze(-1) & in_doc)[0]
                    answer = (lmask & in_doc)[0]
                    correct = (logits.argmax(-1) == probs.argmax(-1))[0] & sup
                    xent = -(probs * torch.log_softmax(logits, -1)).sum(-1)[0] * sup
                    h_row.append(torch.zeros(n_docs, device=self.device).index_add_(0, doc, correct.float()))
                    d_row.append(torch.zeros(n_docs, device=self.device).index_add_(0, doc, answer.float()))
                    c_row.append(torch.zeros(n_docs, device=self.device).index_add_(0, doc, xent))
                    prefix_ok = correct if prefix_ok is None else prefix_ok & correct
                    prefix_valid = answer if prefix_valid is None else prefix_valid & answer
                    chain_hits[step] += prefix_ok.sum()
                    chain_valid[step] += prefix_valid.sum()
                    shift = eagle_core._shift_left_with_zero
                    ids, pmask, probs, lmask = shift(ids), shift(pmask), shift(probs), shift(lmask)
                hits.append(torch.stack(h_row, 1))
                dens.append(torch.stack(d_row, 1))
                ce.append(torch.stack(c_row, 1))
            hits, dens, ce, lengths = torch.cat(hits), torch.cat(dens), torch.cat(ce), torch.cat(lengths)
            acc = (hits / (dens + 1e-6)).mean(0).tolist()          # per dialogue, then the mean (as the authors)
            ploss = (ce / lengths[:, None]).mean(0).tolist()
            chain = (chain_hits / chain_valid.clamp_min(1)).tolist()
            if was_training:
                self.trainer_module.train()
            result = {'step': self.runtime.global_step, 'samples': state['samples'], 'acc': acc, 'ploss': ploss,
                      'tau_estimate': tau_estimate(acc), 'tau_chain': 1 + sum(chain), 'dialogues': len(lengths),
                      'time': time.time()}
            state['val_history'].append(result)
            with open(run_dir / 'val.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            if state['best_val_tau'] is None or result['tau_estimate'] > state['best_val_tau']:
                state['best_val_tau'] = result['tau_estimate']
            log(f'validation at step {result["step"]} ({state["samples"]} dialogues): tau estimate '
                f'{result["tau_estimate"]:.3f}, acc {[round(x, 3) for x in acc]} ({time.time() - t0:.0f} s)')
            self.window['start'] += time.time() - t0               # validation time is not training speed
            return result

        def export(self, name):
            """Draft weights (bf16, NeMo / SGLang key names) + config.json, for the evaluation code."""
            from safetensors.torch import save_file
            out = run_dir / 'milestones' / name
            out.mkdir(parents=True, exist_ok=True)
            draft = self._module().draft_model
            weights = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach()).cpu().contiguous()
                       for k, v in draft.state_dict().items() if not k.startswith('model.embed_tokens')}
            save_file(weights, str(out / 'model.safetensors'))
            config = draft.config.to_dict()
            config.update({'eagle_capture_layers': state['layers'],
                           'eagle_aux_hidden_state_layer_ids': [x - 1 for x in state['layers']],
                           'trained_steps': self.runtime.global_step, 'trained_samples': state['samples']})
            atomic_json(out / 'config.json', config)
            state['milestones'].append(name)
            log(f'exported draft weights to {out}')

    cfg_path = run_dir / 'nemo_config.yaml'
    cfg_path.write_text(json.dumps(recipe_config(args, run_dir, accum), indent=1), encoding='utf-8')  # JSON is YAML
    sys.argv = [sys.argv[0], '-c', str(cfg_path)]
    cfg = recipe.parse_args_and_load_config(None)
    trainer = Ours(cfg)
    recipe.load_or_build_eagle3_token_mapping = lambda loader, *, target_vocab_size, **kw: (
        selected, torch.zeros(target_vocab_size, dtype=torch.bool).index_fill_(0, selected, True))
    torch.manual_seed(S.SEED)                                      # the draft's initialisation, reproducible
    trainer.setup()
    dtypes = sorted({str(p.dtype) for p in trainer.draft_model.parameters() if p.requires_grad})
    if dtypes != ['torch.float32']:
        raise RuntimeError(f'expected fp32 master weights, got {dtypes}')
    if args.diagnose:          # gradient norms per weight for given checkpoints on fixed rows; no training
        diagnose(trainer, args.diagnose.split(','), _pack_collate, torch, args.diagnose_rows)
        return

    # our counters belong to the checkpoint the recipe resumed from (a newer nemo_state.json may be ahead of it)
    resumed = trainer.runtime.global_step
    if resumed:
        ckpts = sorted(Path(trainer.checkpoint_config.checkpoint_dir).glob(f'epoch_*_step_{resumed}/nemo_state.json'))
        saved = json.loads((ckpts[-1] if ckpts else state_path).read_text(encoding='utf-8'))
        if saved.get('step') != resumed:   # the final checkpoint is the recipe's: its counters are in state_path
            raise ValueError(f'No counters saved for the checkpoint at step {resumed}')
        for key in ('train_ids_sha256', 'layers', 'fc_norm', 'rows_per_step', 'smoke', 'master_weights'):
            if saved.get(key) != state[key]:
                raise ValueError(f'Checkpoint at step {resumed} was made with another {key}: {saved[key]} != {state[key]}')
        state.update(saved)
        for name in ('metrics.jsonl', 'val.jsonl'):               # lines after the checkpoint: a dropped past
            keep_until(run_dir / name, resumed)
        log(f'resumed from step {resumed}: {state["samples"]} dialogues, row {resumed * accum} of {len(offsets) - 1}')
    else:
        for name in ('metrics.jsonl', 'val.jsonl'):               # e.g. a rollback before the first checkpoint
            (run_dir / name).unlink(missing_ok=True)
        log(f'new run: ours (NeMo), aux layers {[x - 1 for x in state["layers"]]}, fc_norm {state["fc_norm"]}, '
            f'{len(offsets) - 1} packed rows ({epoch_dialogues} dialogues), {trainer.total_optim_steps} steps')
    if guard['scale'] != 1.0:                                      # after rollbacks: the same schedule, lower
        sched = trainer.lr_scheduler
        sched.base_lrs = [trainer.peak_lr * guard['scale'] for _ in sched.base_lrs]
        for group, base, fn in zip(trainer.optimizer.param_groups, sched.base_lrs, sched.lr_lambdas):
            group['lr'] = base * fn(sched.last_epoch)
        sched._last_lr = [group['lr'] for group in trainer.optimizer.param_groups]
        log(f'lr scale {guard["scale"]} after {len(guard["rollbacks"])} rollback(s): lr now {sched._last_lr[0]:.3g}')
    # skip a step with an abnormal gradient (the recipe clips, then steps: we see the norm in between)
    clip = torch.nn.utils.clip_grad_norm_
    last_norm = {}

    def clip_and_remember(*a, **kw):
        last_norm['value'] = clip(*a, **kw)
        return last_norm['value']

    torch.nn.utils.clip_grad_norm_ = clip_and_remember
    optimizer_step = trainer.optimizer.step

    def guarded_step(*a, **kw):
        norm = last_norm.pop('value', None)
        step = trainer.runtime.global_step + 1                    # the first steps have large norms by nature
        abnormal = norm is not None and step > 50 and not bool(torch.isfinite(norm) & (norm < G['skip_grad_norm']))
        if abnormal or step == force_skip:
            state['skipped_steps'] = state.get('skipped_steps', 0) + 1
            log(f'skipped optimizer step {step}: grad norm {float(norm):.3g}')
            return None
        return optimizer_step(*a, **kw)

    trainer.optimizer.step = guarded_step
    recent = [m['ploss'][0] for m in monitor_jsonl(run_dir / 'metrics.jsonl')[-30:]]
    trainer.best_loss = min(recent) if recent else float('inf')
    trainer.last_checkpoint = time.time()
    trainer.next_val = (state['samples'] // args.val_every + 1) * args.val_every
    trainer.next_milestone = (state['samples'] // args.milestone_every + 1) * args.milestone_every
    trainer.new_window()
    trainer.write_status('starting')
    if not state['val_history']:
        trainer.validate()                                         # the untrained draft: start of the curves
    try:
        trainer.run_train_validation_loop()
    except StopRun as reason:
        if str(reason) != 'diverged, rollbacks exhausted':
            trainer.write_status('stopped')
        log(f'stopped at step {trainer.runtime.global_step} ({reason}; resume later)')
        return
    except Rollback:
        sys.exit(3)
    if not state['val_history'] or state['val_history'][-1]['step'] != trainer.runtime.global_step:
        trainer.validate()
    if 'final' not in state['milestones']:
        trainer.export('final')
    state['step'] = trainer.runtime.global_step
    atomic_json(state_path, state)
    trainer.write_status('done')
    log(f'finished: step {trainer.runtime.global_step}, {state["samples"]} dialogues')


if __name__ == '__main__':
    main()
