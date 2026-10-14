"""Telegram from outside the server: mirror the run from ClearML and serve bot.py + monitor.py from the mirror.

    python relay.py [--project "EAGLE-3 from scratch"] [--mirror runs_mirror] [--every 60]

For servers that reach ClearML but not Telegram (our H200 node). Runs on any machine with internet: needs
ClearML credentials (~/clearml.conf or CLEARML_*) and TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (env or ~/.eagle_secrets).
Every --every seconds it rebuilds, per variant, the files train.py writes on the server (status.json from the task's
user property "status", val.jsonl and metrics.jsonl from the reported scalars, plan.json and reference.json), and
bot.py / monitor.py read them in remote mode (staleness instead of process ids, the server's disk from the status).
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import settings as S


def write(path, text):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def series(scalars, title, name):
    data = scalars.get(title, {}).get(name, {})
    return dict(zip(data.get('x', []), data.get('y', [])))


def mirror_once(project, out):
    from clearml import Task
    tasks, status_tasks = {}, {}
    for t in Task.get_tasks(project_name=project, task_filter={'order_by': ['-last_update']}):
        if t.name in S.VARIANTS and t.name not in tasks:     # the most recently updated task of each variant
            tasks[t.name] = t
        base = t.name[:-len(' status')] if t.name.endswith(' status') else None
        if base in S.VARIANTS and base not in status_tasks:  # clearml_sync.py --status writes only there
            status_tasks[base] = t
    for name in sorted(set(tasks) | set(status_tasks)):
        task = status_tasks.get(name) or tasks[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        props = {k: v.get('value') for k, v in (task.get_user_properties() or {}).items()}
        for key in ('plan', 'reference'):
            if props.get(key):
                write(out / f'{key}.json', props[key])
        if props.get('status'):
            write(d / 'status.json', props['status'])
        if props.get('val_history') and props.get('metrics_tail'):     # compact copies (clearml_sync.py)
            vals = [{'step': st, 'samples': n, 'tau_estimate': tau, 'acc': acc}
                    for st, n, tau, acc in json.loads(props['val_history'])]
            rows = [{'step': r[0], 'ploss': [r[1]], 'tokens_per_s': r[2],
                     **({'samples': r[3], 'acc': [r[4]]} if len(r) > 3 else {})}
                    for r in json.loads(props['metrics_tail'])]
        else:                                                          # older tasks: from the scalars
            scalars = task.get_reported_scalars(max_samples=0)
            tau = series(scalars, 'val tau estimate', 'tau')
            acc = [series(scalars, 'val acc', f'pos{i}') for i in range(7)]
            vals = [{'step': int(s), 'samples': int(s) * S.GLOBAL_BATCH, 'tau_estimate': tau[s],
                     'acc': [a.get(s, 0.0) for a in acc]} for s in sorted(tau)]
            loss = series(scalars, 'train loss', 'pos0')
            speed = series(scalars, 'train', 'tokens_per_s')
            rows = [{'step': int(s), 'ploss': [loss[s]] * 7, 'tokens_per_s': speed.get(s, 0.0)}
                    for s in sorted(loss)]
        write(d / 'val.jsonl', ''.join(json.dumps(v) + '\n' for v in vals))
        write(d / 'metrics.jsonl', ''.join(json.dumps(r) + '\n' for r in rows[-300:]))
    return sorted(set(tasks) | set(status_tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='EAGLE-3 from scratch')
    parser.add_argument('--mirror', default=str(S.HERE / 'runs_mirror'))
    parser.add_argument('--every', type=float, default=60)
    args = parser.parse_args()
    from session import load_secrets
    load_secrets()
    out = Path(args.mirror)
    out.mkdir(parents=True, exist_ok=True)

    import bot
    import monitor
    bot.REMOTE = monitor.REMOTE = True

    def mirror_loop():
        while True:
            try:
                names = mirror_once(args.project, out)
                print(time.strftime('%H:%M:%S'), 'mirrored', names or 'nothing yet', flush=True)
            except Exception as error:          # noqa: BLE001 - keep serving the last copy
                print('mirror failed:', repr(error), flush=True)
            time.sleep(args.every)

    telegram = bot.Bot(os.environ['TELEGRAM_BOT_TOKEN'], out)

    def monitor_loop():
        import dashboard
        notifier, previous = monitor.Notifier(out), {}
        last_summary = time.time()           # the first digest an hour after start (restarts send nothing)
        while True:
            statuses = monitor.check(out, notifier, previous)
            previous = {k: v for k, v in statuses.items() if v}
            if any(statuses.values()) and time.time() - last_summary > 3600:     # the hourly digest: a picture
                try:
                    png = dashboard.render(out)
                    for chat in notifier.chats():
                        telegram.send_photo(chat, png, monitor.summary(out, statuses), html=False)
                except Exception as error:          # noqa: BLE001 - fall back to text
                    print('dashboard failed:', repr(error), flush=True)
                    notifier.send(monitor.summary(out, statuses))
                last_summary = time.time()
            time.sleep(300)

    threading.Thread(target=mirror_loop, daemon=True).start()
    threading.Thread(target=monitor_loop, daemon=True).start()
    telegram.run()


if __name__ == '__main__':
    main()
