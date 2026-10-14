"""Train one draft variant from scratch with the authors' model code; resumable across 8-hour sessions.

    python train.py baseline|ours [--deadline UNIX_TIME] [--data DIR] [--runs DIR] [--total-steps N]

The model, loss and training-time test are the authors' (work/train = patched eagle/traineagle3, see
patch_eagle.py): Model(...) builds the draft and loads the frozen target; its forward returns the 7 TTT losses and
accuracies; the loss is sum(0.8^i * loss_i) as in their main.py. DeepSpeed runs their ds_config.json (fp16 with
loss scaling, ZeRO-2, AdamW (0.9, 0.95), clip 0.5, WarmupDecayLR to 5e-5) with the changes listed in settings.py.

What this loop adds to theirs:
- resumable data order: epoch e is a permutation seeded by (SEED, e); the position inside it is saved;
- checkpoints with the full state (weights, optimizer, loss scaler, lr schedule, data position, RNG) every
  CHECKPOINT_MINUTES, before the session deadline, on SIGTERM / SIGUSR1 and on runs/<variant>/STOP;
- validation on fixed held-out dialogues every VAL_EVERY steps (per-position accuracy and a tau estimate);
- metrics to runs/<variant>/metrics.jsonl, val.jsonl, status.json (read by monitor.py; clearml_sync.py copies
  them to ClearML in its own process, so ClearML can never block training);
- `kill -USR2 <pid>` writes the stack of every thread to train.log (to see where a stuck run waits).
- draft weights exported every MILESTONE_EVERY steps (both variants are compared at equal steps).
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import settings as S


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def tau_estimate(acc):
    """Expected tokens per round of a single draft chain with these per-position accuracies (a proxy for tau)."""
    tau, keep = 1.0, 1.0
    for a in acc:
        keep *= a
        tau += keep
    return tau


def deepspeed_config(total_steps):
    """The authors' ds_config.json with our changes; their values are checked so a surprise fails loudly."""
    cfg = json.loads((S.WORK / 'train' / 'ds_config.json').read_text(encoding='utf-8'))
    opt, sched = cfg['optimizer'], cfg['scheduler']
    expected = (cfg['fp16']['enabled'] in (True, 'true'), opt['type'], opt['params']['betas'],
                opt['params']['weight_decay'], cfg['gradient_clipping'], sched['type'],
                sched['params']['warmup_max_lr'], cfg['zero_optimization']['stage'],
                cfg['train_micro_batch_size_per_gpu'])
    if expected != (True, 'AdamW', [0.9, 0.95], 0.0, 0.5, 'WarmupDecayLR', 5e-5, 2, 1):
        raise ValueError(f'The authors\' ds_config.json differs from what this run assumes: {expected}')
    cfg['gradient_accumulation_steps'] = S.GLOBAL_BATCH            # one GPU, micro-batch 1
    opt['params']['torch_adam'] = True                              # DeepSpeed's fused Adam needs a CUDA toolkit
    sched['params']['total_num_steps'] = total_steps                # theirs: a hard-coded 800000
    sched['params']['warmup_num_steps'] = min(S.WARMUP_STEPS, total_steps // 10)
    cfg['steps_per_print'] = 10 ** 9                                # we log ourselves
    return cfg


def build_model(variant, data, ds_cfg):
    spec = S.VARIANTS[variant]
    os.environ['EAGLE_CAPTURE_LAYERS'] = ','.join(str(i) for i in spec['layers'])   # read by the target model
    sys.path.insert(0, str(S.WORK / 'train'))
    import torch
    from accelerate.utils import set_seed
    set_seed(S.SEED)                                       # as the authors; both variants get the same init
    from cnets import Model
    from configs import EConfig
    config = EConfig.from_pretrained(str(S.WORK / 'train' / 'config.json'))
    config.fc_norm = spec['fc_norm']
    train_config = {'bs': 1, 'num_epochs': 1, 'num_workers': 2, 'max_len': S.MAX_LEN, 'config_path': 'config.json',
                    'gradient_checkpoint': S.GRADIENT_CHECKPOINTING}
    model = Model(config, ds_cfg, train_config, path=str(S.TARGET), load_emb=True, load_head=True)
    # the 32K draft vocabulary of our data, registered as the authors' scandata() would (d2t offsets, t2d mask)
    selected = torch.tensor(json.loads((data / 'vocab.json').read_text(encoding='utf-8')), dtype=torch.long)
    mask = torch.zeros(model.vocab_size, dtype=torch.bool)
    mask[selected] = True
    if selected.numel() != config.draft_vocab_size or int(mask.sum()) != config.draft_vocab_size:
        raise ValueError('Bad draft vocabulary')
    model.register_buffer('d2t', selected - torch.arange(len(selected)))
    model.register_buffer('t2d', mask)
    model.l1smooth = torch.nn.SmoothL1Loss(reduction='none')
    return model, config


class Examples:
    """Single examples of a saved dataset as the authors' collator gives them (micro-batch 1: tensors [1, n])."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        import torch
        row = self.dataset[int(self.indices[k])]
        ids = torch.tensor(row['input_ids'], dtype=torch.long)[None]
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'loss_mask': torch.tensor(row['loss_mask'], dtype=torch.long)[None]}


def epoch_order(n, epoch):
    import torch
    return torch.randperm(n, generator=torch.Generator().manual_seed(S.SEED * 1000 + epoch)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=sorted(S.VARIANTS))
    parser.add_argument('--deadline', type=float, default=0, help='unix time; stop with a checkpoint before it')
    parser.add_argument('--margin-minutes', type=float, default=12)
    parser.add_argument('--data', default=str(S.DATA))
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--total-steps', type=int, default=0, help='default: runs/plan.json')
    parser.add_argument('--log-every', type=int, default=S.LOG_EVERY)
    parser.add_argument('--val-every', type=int, default=S.VAL_EVERY)
    parser.add_argument('--val-eval', type=int, default=S.VAL_EVAL)
    parser.add_argument('--checkpoint-minutes', type=float, default=S.CHECKPOINT_MINUTES)
    parser.add_argument('--milestone-every', type=int, default=S.MILESTONE_EVERY)
    args = parser.parse_args()
    data, runs = Path(args.data), Path(args.runs)
    run_dir = runs / args.variant
    ckpt_dir = run_dir / 'checkpoints'
    for d in (run_dir, ckpt_dir, run_dir / 'milestones'):
        d.mkdir(parents=True, exist_ok=True)
    plan = json.loads((runs / 'plan.json').read_text(encoding='utf-8')) if not args.total_steps else {}
    total_steps = args.total_steps or plan['total_steps']                 # the lr schedule reaches 0 here
    stop_step = plan.get('stop_steps', {}).get(args.variant, total_steps)  # where this variant stops
    compare_step = plan.get('compare_step')                               # both variants exported here
    deadline = args.deadline or float('inf')

    import torch
    import deepspeed
    from datasets import load_from_disk

    stop = {'signal': None}
    if hasattr(signal, 'SIGUSR2'):
        faulthandler.register(signal.SIGUSR2, all_threads=True)      # kill -USR2 <pid>: stacks to train.log
    for name in ('SIGTERM', 'SIGUSR1'):                 # SIGUSR1 does not exist on Windows (local tests)
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda s, f: stop.update(signal=signal.Signals(s).name))

    ds_cfg = deepspeed_config(total_steps)
    model, config = build_model(args.variant, data, ds_cfg)
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                                          config=ds_cfg)
    device = engine.device
    train = load_from_disk(str(data / 'train'))
    val = load_from_disk(str(data / 'val'))
    val_examples = Examples(val, list(range(min(args.val_eval, len(val)))))
    data_stats = json.loads((data / 'stats.json').read_text(encoding='utf-8'))

    state = {'epoch': 0, 'position': 0, 'samples': 0, 'tokens': 0, 'supervised': 0, 'nonfinite': 0,
             'train_seconds': 0.0, 'best_val_tau': None, 'val_history': [], 'total_steps': total_steps,
             'variant': args.variant, 'layers': S.VARIANTS[args.variant]['layers'],
             'fc_norm': S.VARIANTS[args.variant]['fc_norm'], 'train_ids_sha256': data_stats['train_ids_sha256']}
    latest = ckpt_dir / 'latest_ok.json'
    if latest.exists():
        tag = json.loads(latest.read_text(encoding='utf-8'))['tag']
        _, client = engine.load_checkpoint(str(ckpt_dir), tag=tag, load_module_strict=False,
                                           load_optimizer_states=True, load_lr_scheduler_states=True)
        for key in ('total_steps', 'variant', 'layers', 'fc_norm', 'train_ids_sha256'):
            if client[key] != state[key]:
                raise ValueError(f'Checkpoint {tag} was made with another {key}: {client[key]} != {state[key]}')
        rng = client.pop('rng')
        state.update(client)
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python'])
        log(f'resumed from {tag}: step {engine.global_steps}, epoch {state["epoch"]} position {state["position"]}')
    else:
        log(f'new run: {args.variant}, layers {state["layers"]}, fc_norm {state["fc_norm"]}, '
            f'lr schedule over {total_steps} steps, this variant stops at {stop_step}')

    status_path = run_dir / 'status.json'

    def loss_scale():
        opt = engine.optimizer
        scale = getattr(opt, 'loss_scale', None)
        if scale is None and getattr(opt, 'loss_scaler', None) is not None:
            scale = getattr(opt.loss_scaler, 'cur_scale', None)
        return float(scale) if scale is not None else None

    def save(reason):
        tag = f'step{engine.global_steps:07d}'
        client = dict(state, rng={'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
                                  'python': random.getstate()})
        t0 = time.time()
        engine.save_checkpoint(str(ckpt_dir), tag=tag, client_state=client, exclude_frozen_parameters=True)
        atomic_json(latest, {'tag': tag, 'step': engine.global_steps, 'time': time.time(), 'reason': reason})
        for old in sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith('step'))[:-S.KEEP_CHECKPOINTS]:
            shutil.rmtree(old, ignore_errors=True)
        log(f'checkpoint {tag} ({reason}) in {time.time() - t0:.0f} s')
        return time.time()

    def export(name):
        """Draft weights (fp16, the authors' key names) + config with our fields, for the evaluation code."""
        out = run_dir / 'milestones' / name
        out.mkdir(parents=True, exist_ok=True)
        weights = {k: v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu()
                   for k, v in engine.module.state_dict().items()
                   if not k.startswith(('target_model.', 'embed_tokens.')) and not k.endswith('inv_freq')}
        torch.save(weights, out / 'pytorch_model.bin')
        draft_config = json.loads((S.WORK / 'train' / 'config.json').read_text(encoding='utf-8'))
        draft_config.update({'eagle_capture_layers': state['layers'], 'fc_norm': state['fc_norm'],
                             'trained_steps': engine.global_steps, 'trained_samples': state['samples']})
        atomic_json(out / 'config.json', draft_config)
        log(f'exported draft weights to {out}')

    def validate():
        engine.eval()
        n, acc, ploss = 0, [0.0] * model.length, [0.0] * model.length
        with torch.no_grad():
            for ex in val_examples:
                p, _, a = engine(input_ids=ex['input_ids'].to(device), attention_mask=ex['attention_mask'].to(device),
                                 loss_mask=ex['loss_mask'])
                n += 1
                acc = [x + y for x, y in zip(acc, a)]
                ploss = [x + float(y) for x, y in zip(ploss, p)]
        engine.train()
        acc, ploss = [x / n for x in acc], [x / n for x in ploss]
        result = {'step': engine.global_steps, 'samples': state['samples'], 'acc': acc, 'ploss': ploss,
                  'tau_estimate': tau_estimate(acc), 'time': time.time()}
        state['val_history'].append(result)
        with open(run_dir / 'val.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')
        if state['best_val_tau'] is None or result['tau_estimate'] > state['best_val_tau']:
            state['best_val_tau'] = result['tau_estimate']
        log(f'validation at step {engine.global_steps}: tau estimate {result["tau_estimate"]:.3f}, '
            f'acc {[round(x, 3) for x in acc]}')
        return result

    def write_status(phase, window=None):
        status = {
            'variant': args.variant, 'phase': phase, 'pid': os.getpid(), 'updated': time.time(),
            'step': engine.global_steps, 'total_steps': total_steps, 'stop_step': stop_step, 'epoch': state['epoch'],
            'position': state['position'], 'train_dialogues': len(train), 'samples': state['samples'],
            'tokens': state['tokens'], 'nonfinite': state['nonfinite'], 'skipped_steps': int(engine.skipped_steps),
            'lr': float(optimizer.param_groups[0]['lr']), 'loss_scale': loss_scale(), 'window': window,
            'last_val': state['val_history'][-1] if state['val_history'] else None,
            'best_val_tau': state['best_val_tau'], 'last_checkpoint': json.loads(latest.read_text())
            if latest.exists() else None, 'deadline': None if deadline == float('inf') else deadline,
            'train_seconds': state['train_seconds'],
            'gpu_memory_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'disk_free_gb': shutil.disk_usage(run_dir).free / 2 ** 30}
        atomic_json(status_path, status)

    write_status('starting')
    if not state['val_history']:
        validate()                                       # the untrained draft: starting point of the curves
    last_checkpoint = time.time()
    window = {'start': time.time(), 'samples': 0, 'tokens': 0, 'ploss': [0.0] * model.length,
              'acc': [0.0] * model.length, 'micro': 0}
    finished = False
    engine.train()
    while not finished:
        order = epoch_order(len(train), state['epoch'])
        loader = torch.utils.data.DataLoader(Examples(train, order[state['position']:]), batch_size=None,
                                             shuffle=False, num_workers=2, pin_memory=True)
        for ex in loader:
            plosses, _, acces = engine(input_ids=ex['input_ids'].to(device),
                                       attention_mask=ex['attention_mask'].to(device), loss_mask=ex['loss_mask'])
            loss = sum(0.8 ** i * p for i, p in enumerate(plosses))         # the authors' weights
            if not bool(torch.isfinite(loss.detach())):
                state['nonfinite'] += 1           # fp16: DeepSpeed skips the step and lowers the loss scale
            engine.backward(loss)
            engine.step()
            n_tokens = int(ex['input_ids'].shape[1])
            state['position'] += 1
            state['samples'] += 1
            state['tokens'] += n_tokens
            state['supervised'] += int(ex['loss_mask'].sum())
            window['samples'] += 1
            window['tokens'] += n_tokens
            window['micro'] += 1
            window['ploss'] = [x + float(y) for x, y in zip(window['ploss'], plosses)]
            window['acc'] = [x + y for x, y in zip(window['acc'], acces)]
            if state['samples'] % S.GLOBAL_BATCH:
                continue          # decisions only at optimizer-step boundaries (DeepSpeed counts micro-batches globally)
            step = engine.global_steps
            if step % args.log_every == 0:
                elapsed = time.time() - window['start']
                state['train_seconds'] += elapsed
                acc = [x / window['micro'] for x in window['acc']]
                ploss = [x / window['micro'] for x in window['ploss']]
                summary = {'step': step, 'samples': state['samples'], 'epoch': state['epoch'],
                           'tokens_per_s': window['tokens'] / elapsed, 'samples_per_s': window['samples'] / elapsed,
                           'lr': float(optimizer.param_groups[0]['lr']), 'loss_scale': loss_scale(),
                           'skipped_steps': int(engine.skipped_steps), 'nonfinite': state['nonfinite'],
                           'ploss': ploss, 'acc': acc, 'tau_estimate_train': tau_estimate(acc), 'time': time.time()}
                with open(run_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(summary) + '\n')
                write_status('running', summary)
                window = {'start': time.time(), 'samples': 0, 'tokens': 0, 'ploss': [0.0] * model.length,
                          'acc': [0.0] * model.length, 'micro': 0}
            if step % args.val_every == 0 or step == compare_step:
                validate()
            if step % args.milestone_every == 0 or step in (total_steps, stop_step, compare_step):
                export(f'step{step:07d}')
            reason = None
            if step >= stop_step:
                reason, finished = 'plan finished', True
            elif stop['signal']:
                reason, finished = f'signal {stop["signal"]}', True
            elif (run_dir / 'STOP').exists():
                reason, finished = 'STOP file', True
            elif time.time() > deadline - args.margin_minutes * 60:
                reason, finished = 'session deadline', True
            elif (run_dir / 'CKPT').exists():
                reason = 'CKPT file'
                (run_dir / 'CKPT').unlink()
            elif time.time() - last_checkpoint > args.checkpoint_minutes * 60:
                reason = 'periodic'
            if reason:
                last_checkpoint = save(reason)
            if finished:
                break
        else:
            state['epoch'] += 1                           # the loader ran out: next epoch, new order
            state['position'] = 0
    if engine.global_steps >= stop_step and state['val_history'][-1]['step'] != engine.global_steps:
        validate()
    write_status('done' if engine.global_steps >= stop_step else 'stopped')
    log(f'stopped at step {engine.global_steps} ({"plan finished" if engine.global_steps >= stop_step else "resume later"})')


if __name__ == '__main__':
    main()
