"""Read the run from ClearML from anywhere (no server access): last values and a comparison of the variants.

    python clearml_report.py [--project "EAGLE-3 from scratch"]

Needs ClearML credentials on this machine (~/clearml.conf or CLEARML_* variables). Read-only.
"""
import argparse

from clearml import Task


def last_values(task):
    """{title/series: (iteration, value)} of the latest reported scalars."""
    out = {}
    for title, series in task.get_last_scalar_metrics().items():
        for name, v in series.items():
            out[f'{title}/{name}'] = v.get('last')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='EAGLE-3 from scratch')
    args = parser.parse_args()
    tasks = {t.name: t for t in Task.get_tasks(project_name=args.project) if t.name in ('baseline', 'ours')}
    if not tasks:
        print(f'no training tasks in "{args.project}" yet')
        return
    rows = {}
    for name, task in sorted(tasks.items()):
        rows[name] = last_values(task)
        print(f'== {name}: status {task.get_status()}, last iteration {task.get_last_iteration()}')
        for key in ('val tau estimate/tau', 'train/tau_estimate', 'train/tokens_per_s', 'train/lr', 'train/loss_scale',
                    'train/skipped_steps', 'val acc/pos0', 'val acc/pos6', 'train loss/pos0'):
            if key in rows[name]:
                print(f'   {key:24s} {rows[name][key]:.4g}')
    if {'baseline', 'ours'} <= rows.keys():
        b, o = rows['baseline'].get('val tau estimate/tau'), rows['ours'].get('val tau estimate/tau')
        if b is not None and o is not None:
            print(f'== ours - baseline (last validations): {o - b:+.3f} tau estimate')
        hist = {n: tasks[n].get_reported_scalars().get('val tau estimate', {}).get('tau', {}) for n in tasks}
        common = sorted(set(hist['baseline'].get('x', [])) & set(hist['ours'].get('x', [])))
        if common:
            by = {n: dict(zip(hist[n]['x'], hist[n]['y'])) for n in hist}
            print('== at equal steps:', ', '.join(f'{s}: {by["ours"][s] - by["baseline"][s]:+.3f}' for s in common[-6:]))


if __name__ == '__main__':
    main()
