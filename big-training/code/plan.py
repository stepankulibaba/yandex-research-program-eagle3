"""The plan of the run -> runs/plan.json. Written once, before the first real session.

    python plan.py --epoch --compare 0.4           one full epoch; both variants until 0.4 of it, then only "ours"
    python plan.py "2026-10-15 12:00" [--gpu-share 0.8] [--eval-hours 3]     as many steps as fit before a deadline

total_steps is the length of the lr schedule (WarmupDecayLR reaches 0 there). In --epoch mode both variants share
that schedule; baseline stops at the comparison step (both are exported and validated exactly there, a fair
comparison at equal steps and equal lr), ours continues alone to the end of the epoch. Speeds come from the speed test
(runs_speed_solo, runs_speed_conc) or the smoke test; the finish estimate assumes the GPU trains `--gpu-share` of the
wall time. train.py refuses checkpoints made with another total_steps, so the plan must not change mid-run.
"""
import argparse
from datetime import datetime, timedelta
import json
import math
import statistics

import settings as S


def median_speed(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = rows[len(rows) // 3:]                     # skip the warm-up part
    return statistics.median(r['samples_per_s'] for r in rows)


def speeds():
    """(dialogues/s of one variant alone, of each variant when both run side by side)."""
    solo = S.HERE / 'runs_speed_solo' / 'baseline' / 'metrics.jsonl'
    side = [S.HERE / 'runs_speed_conc' / v / 'metrics.jsonl' for v in S.VARIANTS]
    if solo.exists() and all(p.exists() for p in side):
        return median_speed(solo), min(median_speed(p) for p in side)
    side = [S.HERE / 'runs_smoke' / v / 'metrics.jsonl' for v in S.VARIANTS]
    both = min(median_speed(p) for p in side)
    return both * 2.2, both                           # no solo measurement: assume it is ~10 % better than both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deadline', nargs='?', help='deadline mode: "YYYY-MM-DD HH:MM" when training must end')
    parser.add_argument('--epoch', action='store_true', help='one full epoch (lr reaches 0 at its end)')
    parser.add_argument('--compare', type=float, default=1.0, help='epoch mode: baseline stops at this fraction')
    parser.add_argument('--gpu-share', type=float, default=0.85)
    parser.add_argument('--eval-hours', type=float, default=3)
    args = parser.parse_args()
    stats = json.loads((S.DATA / 'stats.json').read_text(encoding='utf-8'))
    solo, side = speeds()
    if args.epoch:
        total = math.ceil(stats['train'] / S.GLOBAL_BATCH)
        compare = round(args.compare * total)
        both_hours = compare * S.GLOBAL_BATCH / side / 3600
        alone_hours = (total - compare) * S.GLOBAL_BATCH / solo / 3600
        finish = datetime.now() + timedelta(hours=(both_hours + alone_hours) / args.gpu_share)
        plan = {'mode': 'epoch', 'total_steps': total, 'compare_step': compare,
                'stop_steps': {'baseline': compare, 'ours': total},
                'share_of_data': 1.0, 'compare_share': args.compare,
                'gpu_hours': {'both': round(both_hours, 1), 'ours_alone': round(alone_hours, 1)},
                'deadline': finish.strftime('%Y-%m-%d %H:%M') + ' (estimate)'}
    else:
        if not args.deadline:
            raise ValueError('give a deadline or --epoch')
        hours = (datetime.strptime(args.deadline, '%Y-%m-%d %H:%M') - datetime.now()).total_seconds() / 3600
        hours -= args.eval_hours
        if hours <= 0:
            raise ValueError('The deadline leaves no time for training')
        total = int(side * hours * args.gpu_share * 3600 // S.GLOBAL_BATCH)
        plan = {'mode': 'deadline', 'total_steps': total, 'share_of_data': total * S.GLOBAL_BATCH / stats['train'],
                'deadline': args.deadline, 'training_hours': hours * args.gpu_share}
    plan.update({'speeds_dialogues_per_s': {'alone': solo, 'both_each': side}, 'gpu_share': args.gpu_share,
                 'made': datetime.now().strftime('%Y-%m-%d %H:%M')})
    S.RUNS.mkdir(parents=True, exist_ok=True)
    path = S.RUNS / 'plan.json'
    if path.exists() and any((S.RUNS / v / 'checkpoints').exists() for v in S.VARIANTS):
        raise ValueError(f'{path}: training has started with it; the plan must not change mid-run')
    path.write_text(json.dumps(plan, indent=1), encoding='utf-8')
    print(json.dumps(plan, indent=1))


if __name__ == '__main__':
    main()
