"""Copies one run's files to ClearML in separate processes, so a stuck ClearML client can never stop training.

    python clearml_sync.py <variant> [--runs DIR] [--every 30]                  the curves (scalars)
    python clearml_sync.py <variant> --status [--runs DIR] [--every 30]         the status for relay.py / Telegram

session.py starts both per variant. Reads what train.py writes (runs/<variant>/metrics.jsonl, val.jsonl,
status.json; runs/plan.json, reference.json) and reports it to the ClearML task "<variant>" of project
CLEARML_PROJECT (default "EAGLE-3 from scratch"). The task id is kept in runs/<variant>/clearml_task_id, so every
session continues the same task. Resource monitoring of the scalars process adds GPU utilisation of this node.

Two processes, because api.clear.ml can time out on the events API for hours (seen 2026-10-12/13): the scalars
client then retries and blocks, and the status (user properties) must still arrive for the Telegram bot. The
--status process does not create a Task session with background reporting, it only edits the task's properties.
"""
import argparse
import json
import os
from pathlib import Path
import time

import settings as S


def lines_from(path, offset):
    """New complete lines of a growing file after byte offset -> (records, new offset)."""
    if not path.exists():
        return [], offset
    if path.stat().st_size < offset:       # the trainer rewrote the file (a rollback dropped its newest lines)
        offset = 0
    with open(path, 'rb') as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b'\n') + 1
    records = [json.loads(x) for x in chunk[:end].decode('utf-8').splitlines() if x.strip()]
    return records, offset + end


def complete(path):
    """All complete records of a .jsonl file (the line being written is skipped)."""
    return lines_from(path, 0)[0]


def status_properties(runs, run_dir):
    props = {}
    for name, path in (('status', run_dir / 'status.json'), ('plan', runs / 'plan.json'),
                       ('reference', runs / 'reference.json')):
        if path.exists():
            props[name] = path.read_text(encoding='utf-8')
    # the curves for relay.py as plain properties too: they arrive at once, unlike buffered scalars
    history = complete(run_dir / 'val.jsonl')
    props['val_history'] = json.dumps([[v['step'], v['samples'], round(v['tau_estimate'], 4),
                                        [round(a, 4) for a in v['acc']]] for v in history])
    tail = complete(run_dir / 'metrics.jsonl')[-300:]
    props['metrics_tail'] = json.dumps([[m['step'], round(m['ploss'][0], 4), round(m['tokens_per_s']),
                                         m.get('samples'), round(m['acc'][0], 4)] for m in tail])
    return props


def run_status(args, runs, run_dir):
    """The status goes to its own task "<variant> status": the scalars process holds a cached copy of the main
    task's properties and sometimes writes it back, which put a stale status (a finished session's) in front of
    the bot (2026-10-13 13:38)."""
    from clearml import Task
    project = os.environ.get('CLEARML_PROJECT', 'EAGLE-3 from scratch')
    id_file = run_dir / 'clearml_status_task_id'
    task = None
    if id_file.exists():
        try:
            task = Task.get_task(task_id=id_file.read_text().strip())
        except Exception:                   # noqa: BLE001 - deleted on the server: make a new one
            task = None
    if task is None:
        task = Task.create(project_name=project, task_name=f'{args.variant} status', task_type='monitor')
        id_file.write_text(task.id)
    while True:
        try:
            task.set_user_properties(**status_properties(runs, run_dir))
        except Exception as error:          # noqa: BLE001 - try again next round
            print(time.strftime('%H:%M:%S'), 'user properties failed:', repr(error), flush=True)
        if args.until and time.time() > args.until:
            break
        time.sleep(args.every)


def run_scalars(args, runs, run_dir):
    from clearml import Task
    id_file = run_dir / 'clearml_task_id'
    previous = id_file.read_text().strip() if id_file.exists() else False
    task = Task.init(project_name=os.environ.get('CLEARML_PROJECT', 'EAGLE-3 from scratch'), task_name=args.variant,
                     continue_last_task=previous or False, reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False, auto_resource_monitoring=True)
    id_file.write_text(task.id)
    if S.VARIANTS[args.variant]['trainer'] == 'nemo':
        protocol = {**S.NEMO_RECIPE, 'bf16': True, 'compile': S.NEMO_COMPILE}
    else:
        protocol = {'global_batch': S.GLOBAL_BATCH, 'lr': 5e-5, 'fp16': True}
    task.connect({'variant': args.variant, **S.VARIANTS[args.variant], 'dataset': S.DATASET,
                  'dataset_revision': S.DATASET_REVISION, **protocol}, name='run')
    logger = task.get_logger()
    offset_file = run_dir / 'clearml_offsets.json'     # what this task already has (bytes read per file)
    offsets = {'metrics': 0, 'val': 0, 'task': task.id}
    if offset_file.exists():
        saved = json.loads(offset_file.read_text())
        if saved.get('task') == task.id:
            offsets = saved
    while True:
        try:
            metrics, offsets['metrics'] = lines_from(run_dir / 'metrics.jsonl', offsets['metrics'])
            for m in metrics:
                step = m['step']
                for i, (loss, acc) in enumerate(zip(m['ploss'], m['acc'])):
                    logger.report_scalar('train loss', f'pos{i}', loss, iteration=step)
                    logger.report_scalar('train acc', f'pos{i}', acc, iteration=step)
                for key in ('tau_estimate_train', 'tau_sim_train', 'tokens_per_s', 'samples_per_s', 'samples', 'lr',
                            'grad_norm', 'loss_scale', 'skipped_steps', 'nonfinite'):
                    if m.get(key) is not None:
                        logger.report_scalar('train', key, m[key], iteration=step)
            vals, offsets['val'] = lines_from(run_dir / 'val.jsonl', offsets['val'])
            for v in vals:
                logger.report_scalar('val tau estimate', 'tau', v['tau_estimate'], iteration=v['step'])
                for i, (acc, loss) in enumerate(zip(v['acc'], v['ploss'])):
                    logger.report_scalar('val acc', f'pos{i}', acc, iteration=v['step'])
                    logger.report_scalar('val loss', f'pos{i}', loss, iteration=v['step'])
            logger.flush()
            offset_file.write_text(json.dumps(offsets))
        except Exception as error:          # noqa: BLE001
            print(time.strftime('%H:%M:%S'), 'scalars failed:', repr(error), flush=True)
        if args.until and time.time() > args.until:
            break
        time.sleep(args.every)
    task.flush(wait_for_uploads=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=sorted(S.VARIANTS))
    parser.add_argument('--status', action='store_true', help='only the status properties (for relay.py)')
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--every', type=float, default=30)
    parser.add_argument('--until', type=float, default=0)
    args = parser.parse_args()
    runs = Path(args.runs)
    run_dir = runs / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_status if args.status else run_scalars)(args, runs, run_dir)


if __name__ == '__main__':
    main()
