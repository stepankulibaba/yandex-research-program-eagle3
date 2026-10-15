"""Watches both runs: alerts at once, a summary every hour (Telegram if configured, always runs/monitor.log).

    python monitor.py [--runs DIR] [--every 300] [--summary-every 3600] [--once]
    env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (optional; ~/.eagle_secrets is read by session.py)

Alerts (each repeated at most once an hour): a run stalls (no new step for 20 min) or its process died; fp16
trouble (loss scale at its minimum, many skipped steps, non-finite losses); the validation tau estimate falls well
below its best; less than 30 GB free on disk.
The summary has, per variant: step / plan, samples, speed, time to the end of the plan, the last validation, and a
forecast: a saturating curve tau(samples) = a - b * samples^(-c) fitted to the validation points, evaluated at the
end of the plan and after a full pass over the data; with runs/reference.json (the official draft scored on the
same validation dialogues) also how many samples the fit needs to reach it.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

import settings as S

STALL_MINUTES = 20
REMOTE = False        # relay.py: data mirrored from ClearML, so no process ids and no local disk to look at


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def jsonl(path):
    """Records of a growing .jsonl file: complete lines only (a line being written is left for the next read)."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    out = []
    for line in text[:text.rfind('\n') + 1].splitlines():
        try:
            out.append(json.loads(line)) if line.strip() else None
        except ValueError:
            pass
    return out


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def fit_curve(points):
    """Least squares of tau = a - b * x^(-c) over a grid of c; None with fewer than 4 points."""
    import numpy as np
    pts = [(x, y) for x, y in points if x > 0]
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    best = None
    for c in np.linspace(0.05, 1.5, 59):
        design = np.stack([np.ones_like(x), -x ** (-c)], axis=1)
        (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(((design @ [a, b] - y) ** 2).sum())
        if b > 0 and (best is None or sse < best[3]):
            best = (float(a), float(b), float(c), sse)
    return best


def predict(fit, samples):
    a, b, c, _ = fit
    return a - b * samples ** (-c)


def samples_to_reach(fit, level):
    a, b, c, _ = fit
    return None if level >= a else (b / (a - level)) ** (1 / c)


class Notifier:
    def __init__(self, runs):
        self.log = runs / 'monitor.log'
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.owner = [c.strip() for c in os.environ.get('TELEGRAM_CHAT_ID', '').split(',') if c.strip()]
        self.sent = {}

    def chats(self):
        return list(self.owner)            # colleagues only look at the bot; messages go to the owner

    def send(self, text, key=None):
        if key is not None:
            if time.time() - self.sent.get(key, 0) < 3600:
                return
            self.sent[key] = time.time()
        with open(self.log, 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n\n')
        if not self.token:
            return
        for chat in self.chats():
            try:
                body = urllib.parse.urlencode({'chat_id': chat, 'text': text[:4000]}).encode()
                urllib.request.urlopen(f'https://api.telegram.org/bot{self.token}/sendMessage', body, timeout=20)
            except OSError as error:
                with open(self.log, 'a', encoding='utf-8') as f:
                    f.write(f'telegram failed for {chat}: {error!r}\n')


NAMES = {'baseline': '📄 baseline', 'ours': '🧪 ours'}


def progress(st):
    """(dialogues trained, dialogues at this variant's stop, hours left): steps differ between the trainers
    (baseline 8 dialogues per step, ours 32 packed rows), dialogues do not."""
    stop = st.get('stop_samples') or st.get('stop_step', st['total_steps']) * S.GLOBAL_BATCH
    done = st.get('samples', st['step'] * S.GLOBAL_BATCH)
    speed = (st.get('window') or {}).get('samples_per_s')
    return done, stop, (stop - done) / speed / 3600 if speed else None


def compare(runs):
    """ours - baseline in tau estimate at an equal number of trained dialogues, or None.

    The point is the newest validation of either variant that the other one's history covers; the other's tau
    there is interpolated linearly between its two neighbouring validations."""
    hist = {v: [h for h in val_history(runs / v)] for v in ('baseline', 'ours')}
    if not all(hist.values()):
        return None

    def at(points, x):
        for a, b in zip(points, points[1:]):
            if a['samples'] <= x <= b['samples']:
                t = (x - a['samples']) / max(1, b['samples'] - a['samples'])
                return a['tau_estimate'] + t * (b['tau_estimate'] - a['tau_estimate'])
        return points[-1]['tau_estimate'] if points[-1]['samples'] == x else None

    best = None
    for v, other in (('ours', 'baseline'), ('baseline', 'ours')):
        for h in reversed(hist[v]):
            tau = at(hist[other], h['samples'])
            if tau is not None and h['samples'] > 0:
                diff = h['tau_estimate'] - tau if v == 'ours' else tau - h['tau_estimate']
                if best is None or h['samples'] > best[1]:
                    best = (diff, h['samples'])
                break
    return best


def compare_line(runs, html=False):
    c = compare(runs)
    if c is None:
        return None
    diff, samples = c
    icon = '🟢' if diff > 0.02 else ('🔴' if diff < -0.02 else '🟡')
    text = f'ours − baseline: {diff:+.2f} τ'
    return f'{icon} ' + (f'<b>{text}</b>' if html else text) + f' (при {samples / 1e3:,.0f} тыс. диалогов)'.replace(',', ' ')


def hours_left(statuses, plan=None):
    """GPU hours left per variant, counting that the one which finishes first frees the GPU: from then on the other
    runs at the solo speed (plan.json speeds_dialogues_per_s.alone; measured 9.7 vs 4.4 dialogues/s side by side)."""
    plan = plan or {}
    speeds = plan.get("speeds_dialogues_per_s", {})
    left = {}
    for v, st in statuses.items():
        if not st or st.get("phase") == "done":
            continue
        done, stop, _ = progress(st)
        if stop - done <= 0:
            continue
        speed = (st.get("window") or {}).get("samples_per_s") or (speeds.get("both_each") if st["phase"] != "done" else None)
        if speed:
            left[v] = (stop - done, speed)
    out = {v: rem / sp / 3600 for v, (rem, sp) in left.items()}
    if len(left) == 2:
        (a, (ra, sa)), (b, (rb, sb)) = sorted(left.items(), key=lambda kv: kv[1][0] / kv[1][1])
        t1 = ra / sa                                             # seconds until the first one stops
        solo = speeds.get("alone") or sb * 2.2
        out[a] = t1 / 3600
        out[b] = (t1 + (rb - sb * t1) / solo) / 3600
    return out


def bar(fraction, width=10):
    full = round(max(0.0, min(1.0, fraction)) * width)
    return '▰' * full + '▱' * (width - full)


def check(runs, notifier, previous):
    """Alerts for every variant (in Russian, to the owner only); returns the statuses."""
    statuses = {}
    for variant in S.VARIANTS:
        st = read(runs / variant / 'status.json')
        statuses[variant] = st
        if st is None:
            continue
        name, prev = NAMES[variant], previous.get(variant) or st
        age = (time.time() - st['updated']) / 60
        if REMOTE:
            if st['phase'] in ('starting', 'running') and age > STALL_MINUTES:
                notifier.send(f'🟠 {name}: с сервера нет вестей {age:.0f} мин (шаг {st["step"]}).\n'
                              f'Завис, упал или сессия кончилась без сохранения — проверь сервер.', key=f'stall-{name}')
        elif st['phase'] in ('starting', 'running') and not alive(st['pid']):
            notifier.send(f'🔴 {name}: процесс обучения пропал (шаг {st["step"]}).\n'
                          f'Лог: runs/{variant}/train.log', key=f'dead-{name}')
        elif st['phase'] in ('starting', 'running') and age > STALL_MINUTES:
            notifier.send(f'🟠 {name}: нет нового шага {age:.0f} мин (шаг {st["step"]})', key=f'stall-{name}')
        deadline = st.get('deadline')
        if deadline and st['phase'] in ('starting', 'running'):
            left = (deadline - time.time()) / 60
            for mark in (60, 20):                 # one warning at each mark per session
                if 0 < left <= mark:
                    notifier.send(f'⏰ Сессия закончится через {left:.0f} мин. Обучение само сохранится за 12 мин '
                                  f'до конца — приготовься запустить новую сессию.', key=f'end{mark}-{int(deadline)}')
                    break
        if st['phase'] == 'stopped' and progress(st)[0] < progress(st)[1]:
            notifier.send('🔁 Сессия закончилась, обучение сохранено и ждёт (шаг '
                          f'{st["step"]:,}). Запусти новую сессию и в ней:\n'.replace(',', ' ')
                          + 'cd ~/data/big_training && nohup setsid bash session.sh > session.out 2>&1 &',
                          key='restart')
        rolls = st.get('rollbacks') or []
        if len(rolls) > len(prev.get('rollbacks') or []):          # train_nemo_full.py's divergence guard
            r = rolls[-1]
            notifier.send(f'🟠 {name}: обучение разошлось на шаге {r["step"]:,} ({r["why"]}).\n'.replace(',', ' ')
                          + f'Откат к последнему здоровому чекпоинту, lr ×{st.get("lr_scale", 0):g} '
                          f'(откатов: {len(rolls)} из 3).', key=f'rollback-{r["step"]}-{len(rolls)}')
        if st.get('loss_scale') is not None and st['loss_scale'] <= 1:
            notifier.send(f'🟠 {name}: fp16 loss scale упал до {st["loss_scale"]:g}', key=f'scale-{name}')
        if st['skipped_steps'] - prev['skipped_steps'] > 50:
            notifier.send(f'🟠 {name}: пропущено {st["skipped_steps"] - prev["skipped_steps"]} шагов '
                          f'(переполнение fp16) с прошлой проверки', key=f'skip-{name}')
        if st['nonfinite'] - prev['nonfinite'] > 20:
            notifier.send(f'🟠 {name}: {st["nonfinite"] - prev["nonfinite"]} раз loss был NaN/inf '
                          f'с прошлой проверки', key=f'nan-{name}')
        val = st.get('last_val')
        if val and st.get('best_val_tau') and val['tau_estimate'] < st['best_val_tau'] - 0.15:
            notifier.send(f'🟠 {name}: τ на валидации упала до {val["tau_estimate"]:.2f} '
                          f'(лучшее было {st["best_val_tau"]:.2f})', key=f'val-{name}')
    if REMOTE:
        free = min((st.get('disk_free_gb', 1e9) for st in statuses.values() if st), default=1e9)
    else:
        free = shutil.disk_usage(runs).free / 2 ** 30
    if free < 30:
        notifier.send(f'🟠 на диске сервера осталось {free:.0f} ГБ', key='disk')
    return statuses


def summary(runs, statuses):
    """The hourly digest: short, one block per variant, then the comparison and the forecast."""
    reference = read(runs / 'reference.json')
    lines = [f'🦅 EAGLE-3 · сводка {time.strftime("%H:%M")}', '']
    hours = hours_left(statuses, read(runs / 'plan.json'))   # counts the speed-up once one variant stops
    vals, forecasts = {}, {}
    for variant, st in statuses.items():
        if st is None:
            lines += [f'{NAMES[variant]} — ещё не начинался', '']
            continue
        done, stop, _ = progress(st)                      # baseline stops at the comparison point
        left = hours.get(variant)
        frac = done / max(1, stop)
        w = st.get('window') or {}
        speed = w.get('samples_per_s')
        val = st.get('last_val')
        state = {'done': '✅', 'running': '🟢', 'starting': '🟡'}.get(st['phase'], '⏸')
        lines.append(f'{NAMES[variant]}  {state}')
        lines.append(f'{bar(frac)} {frac:.0%}  ·  {done / 1e3:,.0f} из {stop / 1e3:,.0f} тыс. диалогов'.replace(',', ' '))
        detail = [f'τ ≈ {val["tau_estimate"]:.2f}' if val else 'τ — пока нет']
        if speed:
            detail.append(f'{w["tokens_per_s"] / 1000:.1f}k ток/с')
        if left is not None:
            detail.append(f'осталось ~{left:.1f} ч' if left >= 1 else f'осталось ~{left * 60:.0f} мин')
        lines += ['  ·  '.join(detail), '']
        if val:
            vals[variant] = val
        hist = val_history(runs / variant)
        fit = fit_curve([(v['samples'], v['tau_estimate']) for v in hist]) if hist else None
        if fit:
            forecasts[variant] = predict(fit, stop)
    if compare_line(runs):
        lines.append(compare_line(runs))
    if forecasts:
        lines.append('🔮 к концу плана: ' + ' · '.join(f'{NAMES[v].split()[1]} {x:.2f}' for v, x in forecasts.items()))
    if reference:
        lines.append(f'🏁 официальный драфт: {reference["tau_estimate"]:.2f}')
    return '\n'.join(lines).rstrip()


def val_history(run_dir):
    """All validations of a run (train.py appends one line per validation to val.jsonl)."""
    path = run_dir / 'val.jsonl'
    if not path.exists():
        return []
    return jsonl(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--every', type=float, default=300)
    parser.add_argument('--summary-every', type=float, default=3600)
    parser.add_argument('--until', type=float, default=0, help='unix time to stop (session end)')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    notifier = Notifier(runs)
    previous, last_summary = {}, 0
    while True:
        statuses = check(runs, notifier, previous)
        previous = {k: v for k, v in statuses.items() if v}
        if args.once or time.time() - last_summary > args.summary_every:
            notifier.send(summary(runs, statuses))
            last_summary = time.time()
        if args.once or (args.until and time.time() > args.until):
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
