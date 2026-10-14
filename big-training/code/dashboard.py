"""One PNG with the state of the run, for Telegram: progress, tau curves with the forecast, loss, accuracy by depth.

    python dashboard.py [--runs DIR] [--out dashboard.png]

Reads the same files as monitor.py (runs/<variant>/status.json, val.jsonl, metrics.jsonl, runs/plan.json,
reference.json). bot.py sends it on 📊; relay.py attaches it to the hourly digest.
"""
import argparse
import io
from pathlib import Path
import time

import monitor
import settings as S

COLORS = {'baseline': '#8C96A3', 'ours': '#2F6DB5'}
LABELS = {'baseline': 'baseline · слои статьи', 'ours': 'ours · наши слои + fc_norm'}


def metrics(run_dir):
    import json
    path = run_dir / 'metrics.jsonl'
    if not path.exists():
        return []
    return monitor.jsonl(path)


def smooth(values, k=5):
    out, acc = [], []
    for v in values:
        acc.append(v)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def render(runs):
    """PNG bytes of the dashboard."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    statuses = {v: monitor.read(runs / v / 'status.json') for v in S.VARIANTS}
    plan = monitor.read(runs / 'plan.json') or {}
    reference = monitor.read(runs / 'reference.json')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), gridspec_kw={'height_ratios': [1, 1.4]})
    fig.suptitle(f'EAGLE-3 с нуля · {time.strftime("%d.%m %H:%M")}', fontweight='bold', fontsize=14)

    # 1. progress in dialogues (the trainers' steps differ): each variant to its own stop; the comparison marked
    ax = axes[0, 0]
    epoch = max([(s or {}).get('train_dialogues', 0) for s in statuses.values()] + [1])
    total = max(epoch, (plan.get('total_steps') or 0) * S.GLOBAL_BATCH)
    compare = (plan.get('compare_step') or 0) * S.GLOBAL_BATCH or None
    hours = monitor.hours_left(statuses, plan)
    for i, v in enumerate(reversed(list(S.VARIANTS))):     # baseline on top
        st = statuses[v]
        step, stop, _ = monitor.progress(st) if st else (0, plan.get('stop_steps', {}).get(v, 0) * S.GLOBAL_BATCH, None)
        left = hours.get(v)
        ax.barh(i, stop, color='#E6E9ED', height=0.55)
        ax.barh(i, step, color=COLORS[v], height=0.55)
        note = (f'{step / 1e3:,.0f}k / {stop / 1e3:,.0f}k диалогов'.replace(',', ' ')
                + (f'  · ~{left:.0f} ч GPU' if left and left >= 1 else ''))
        ax.text(0, i + 0.38, f'{v}  {note}', va='bottom', fontsize=10)
    if compare:
        ax.axvline(compare, color='#B23A48', ls='--', lw=1)
        ax.text(compare, -0.55, 'сравнение', color='#B23A48', ha='center', va='top', fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.8, len(S.VARIANTS) - 0.1)
    ax.set_xticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title('Прогресс', loc='left')

    # 2. accuracy by draft depth at the last validation
    ax = axes[0, 1]
    width = 0.38
    for i, v in enumerate(S.VARIANTS):
        val = (statuses[v] or {}).get('last_val')
        if val:
            xs = [k + (i - 0.5) * width for k in range(len(val['acc']))]
            ax.bar(xs, val['acc'], width, color=COLORS[v], label=LABELS[v].split(' ·')[0])
    if reference and reference.get('acc'):
        ax.plot(range(len(reference['acc'])), reference['acc'], 'o--', color='#2E7D4F', ms=4, label='официальный')
    ax.set_xticks(range(7), [str(k + 1) for k in range(7)])
    ax.set_xlabel('шаг драфта')
    ax.set_ylim(0, 1)
    ax.set_title('Точность по шагам драфта', loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper right')

    # 3. validation tau estimate vs dialogues, with the fitted forecast and the official draft's level
    ax = axes[1, 0]
    for v in S.VARIANTS:
        hist = [h for h in monitor.val_history(runs / v) if h['samples'] > 0]
        if not hist:
            continue
        xs = [h['samples'] / 1e3 for h in hist]
        ax.plot(xs, [h['tau_estimate'] for h in hist], 'o-', color=COLORS[v], ms=3, label=LABELS[v].split(' ·')[0])
        fit = monitor.fit_curve([(h['samples'], h['tau_estimate']) for h in hist])
        st = statuses[v]
        if fit and st:
            end = monitor.progress(st)[1]
            grid = [hist[-1]['samples'] + (end - hist[-1]['samples']) * k / 30 for k in range(31)]
            ax.plot([g / 1e3 for g in grid], [monitor.predict(fit, g) for g in grid], ':', color=COLORS[v], lw=1.5)
    if reference:
        ax.axhline(reference['tau_estimate'], color='#2E7D4F', ls='--', lw=1)
        ax.text(0, reference['tau_estimate'], ' официальный драфт', color='#2E7D4F', va='bottom',
                fontsize=9)
    if compare:
        ax.axvline(compare / 1e3, color='#B23A48', ls='--', lw=1)
    ax.set_xlim(0, total / 1e3)
    ax.set_ylim(bottom=1)
    ax.set_xlabel('диалогов, тыс.')
    ax.set_title('τ-оценка: валидация и прогноз ┄', loc='left')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    # 4. training accuracy of draft step 1 (correct / answer positions; baseline averages it per dialogue, ours per
    #    token over the window: close, not identical; their losses are defined differently, so no loss panel)
    ax = axes[1, 1]
    for v in S.VARIANTS:
        rows = [r for r in metrics(runs / v) if r.get('samples') is not None]
        if rows:
            ax.plot([r['samples'] / 1e3 for r in rows], smooth([r['acc'][0] for r in rows], 9), color=COLORS[v],
                    lw=1.5, label=LABELS[v].split(' ·')[0])
    ax.set_xlabel('диалогов, тыс.')
    ax.set_xlim(left=0)
    ax.set_title('Точность шага 1 на обучении (ориентир)', loc='left')
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return buf.getvalue()


def caption(runs):
    """Two or three short lines to go with the picture."""
    statuses = {v: monitor.read(runs / v / 'status.json') for v in S.VARIANTS}
    plan = monitor.read(runs / 'plan.json') or {}
    parts = []
    for v, st in statuses.items():
        if not st:
            parts.append(f'{v}: ещё не начинался')
            continue
        done, stop, _ = monitor.progress(st)
        icon = {'done': '✅', 'running': '🟢', 'starting': '🟡'}.get(st['phase'], '⏸')
        val = st.get('last_val')
        parts.append(f'{icon} {v}: {done / max(1, stop):.0%}' + (f' · τ {val["tau_estimate"]:.2f}' if val else ''))
    lines = ['   '.join(parts)]
    if monitor.compare_line(runs):                         # at equal dialogues
        lines.append(monitor.compare_line(runs))
    if plan.get('deadline'):
        lines.append(f'финиш ~{plan["deadline"].split(" (")[0]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--out', default='dashboard.png')
    args = parser.parse_args()
    Path(args.out).write_bytes(render(Path(args.runs)))
    print(caption(Path(args.runs)))
