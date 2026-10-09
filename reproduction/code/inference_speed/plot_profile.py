"""Pictures of the profile: where the GPU waits for the CPU.

    venv_eagle/bin/python plot_profile.py [results/main/profile]     (after profile_eagle.py; needs matplotlib)

Reads trace_eagle.json and trace_plain.json (torch.profiler, Chrome trace format) and draws:
  timeline.png   two EAGLE rounds and three plain steps: phases on the CPU, kernel launches, kernels on the GPU
                 (each kernel coloured by the phase that launched it); the white gaps on the GPU lane are idle time
  gpu_busy.png   share of time the GPU computes, in 2 ms bins over the whole answer
  kernels.png    how long the kernels are (most take microseconds) and which ones take the GPU time
Plus stats.json with the numbers behind them.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'draft': '#2F6DB5', 'verify': '#C2703D', 'target forward': '#C2703D', 'accept': '#7A7A7A',
          'update': '#A8A8A8', 'prefill': '#5B8C5A', None: '#D0D0D0'}
NAMES = {'draft': 'драфт', 'verify': 'проверка дерева', 'target forward': 'прогон большой модели',
         'accept': 'выбор ветки', 'update': 'обновление', 'prefill': 'prefill'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})


def load(path):
    """Kernels, kernel launches and phase spans from a Chrome trace; times in microseconds from the start."""
    events = json.loads(Path(path).read_text(encoding='utf-8'))['traceEvents']
    kernels, launches, phases = [], {}, []
    for e in events:
        if e.get('ph') != 'X':
            continue
        cat, args = e.get('cat', ''), e.get('args', {})
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            kernels.append((float(e['ts']), float(e['dur']), e['name'], args.get('correlation')))
        elif cat == 'cuda_runtime' and 'Launch' in e['name']:
            launches[args.get('correlation')] = (float(e['ts']), float(e['dur']))
        elif cat == 'user_annotation' and e['name'].startswith('phase: '):
            phases.append((float(e['ts']), float(e['dur']), e['name'][len('phase: '):]))
    t0 = min(k[0] for k in kernels)
    kernels = sorted((ts - t0, dur, name, corr) for ts, dur, name, corr in kernels)
    launches = {c: (ts - t0, dur) for c, (ts, dur) in launches.items()}
    # Innermost phase wins (draft runs inside update): sort so longer spans come first, later ones override.
    phases = sorted(((ts - t0, dur, label) for ts, dur, label in phases), key=lambda p: -p[1])
    return kernels, launches, phases


def phase_at(phases, t):
    found = None
    for ts, dur, label in phases:            # longest first, so the last match is the innermost
        if ts <= t <= ts + dur:
            found = label
    return found


def busy_share(kernels, start, end):
    """Share of [start, end] covered by at least one kernel."""
    covered, last = 0.0, start
    for ts, dur, _, _ in kernels:
        a, b = max(ts, last), min(ts + dur, end)
        if b > a:
            covered += b - a
            last = b
    return covered / (end - start)


def timeline_window(phases, label, steps):
    starts = sorted(ts for ts, _, lab in phases if lab == label)
    k = len(starts) // 2
    return starts[k] - 200, starts[min(k + steps, len(starts) - 1)]


def zoom_window(phases, label, width_us=500):
    """A short window in the middle of a typical `label` span, to see single kernels and the gaps between them."""
    spans = sorted((ts, dur) for ts, dur, lab in phases if lab == label)
    ts, dur = spans[len(spans) // 2]
    center = ts + dur / 2
    return center - width_us / 2, center + width_us / 2


def draw_timeline(ax, kernels, launches, phases, window, title):
    lo, hi = window
    for ts, dur, label in phases:            # longest first: inner phases (draft inside update) are drawn on top
        if ts + dur >= lo and ts <= hi:
            ax.broken_barh([(ts / 1000, dur / 1000)], (2.6, 0.8), color=COLORS.get(label), alpha=0.9)
    ticks = [ts for ts, _ in launches.values() if lo <= ts <= hi]
    ax.vlines(np.array(ticks) / 1000, 1.45, 2.25, color='#555555', lw=0.25)
    by_phase = {}
    for ts, dur, _, corr in kernels:
        if lo <= ts <= hi:
            launch = launches.get(corr)
            label = phase_at(phases, launch[0]) if launch else None
            by_phase.setdefault(label, []).append((ts / 1000, max(dur, 1) / 1000))
    for label, spans in by_phase.items():
        ax.broken_barh(spans, (0.2, 0.9), color=COLORS.get(label, COLORS[None]))
    share = busy_share(kernels, lo, hi)
    ax.set_yticks([0.65, 1.85, 3.0], ['GPU: ядра', 'CPU: запуски ядер', 'CPU: фаза'])
    ax.set_xlim(lo / 1000, hi / 1000)
    ax.set_title(f'{title}: GPU занят {100 * share:.0f}% времени окна, '
                 f'{sum(1 for t in ticks)} запусков ядер', fontsize=9, loc='left')
    return share


def gpu_busy_curve(kernels, bin_ms=2.0):
    end = max(ts + dur for ts, dur, _, _ in kernels)
    edges = np.arange(0, end + bin_ms * 1000, bin_ms * 1000)
    busy = np.zeros(len(edges) - 1)
    for ts, dur, _, _ in kernels:
        a, b = ts, ts + dur
        i = int(a // (bin_ms * 1000))
        while a < b and i < len(busy):
            right = edges[i + 1]
            busy[i] += min(b, right) - a
            a = right
            i += 1
    return edges[:-1] / 1000, np.minimum(busy / (bin_ms * 1000), 1.0)


def short(name):
    return name.split('<')[0].split('(')[0].replace('void ', '')[:48]


def main(folder='results/main/profile'):
    folder = Path(folder)
    out = folder / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    eagle = load(folder / 'trace_eagle.json')
    plain = load(folder / 'trace_plain.json')
    stats = {}

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    stats['eagle_window_busy'] = draw_timeline(axes[0], *eagle, timeline_window(eagle[2], 'verify', 2),
                                               'EAGLE-3, два цикла')
    stats['draft_zoom_busy'] = draw_timeline(axes[1], *eagle, zoom_window(eagle[2], 'draft'),
                                             'Увеличение: 0.5 мс внутри драфта')
    stats['verify_zoom_busy'] = draw_timeline(axes[2], *eagle, zoom_window(eagle[2], 'verify'),
                                              'Увеличение: 0.5 мс внутри проверки дерева')
    stats['plain_window_busy'] = draw_timeline(axes[3], *plain, timeline_window(plain[2], 'target forward', 3),
                                               'Обычная генерация, три шага')
    axes[3].set_xlabel('мс (под профилировщиком: всё медленнее, чем без него, пропорции те же)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ('verify', 'draft', 'accept', 'update')]
    fig.legend(handles, [NAMES['verify'] + ' / прогон большой модели', NAMES['draft'], NAMES['accept'], NAMES['update']],
               loc='upper center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle('Таймлайн: белые промежутки на дорожке GPU — простой, пока CPU запускает следующие ядра',
                 fontweight='bold', y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'timeline.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    for (kernels, _, _), label, color in ((eagle, 'EAGLE-3', COLORS['draft']), (plain, 'обычная генерация', COLORS['verify'])):
        t, busy = gpu_busy_curve(kernels)
        ax.plot(t, 100 * busy, lw=0.6, color=color, label=f'{label}: в среднем {100 * busy.mean():.0f}%')
        stats[f'{label} mean busy'] = float(busy.mean())
    ax.set_ylim(0, 105)
    ax.set_ylabel('GPU считает, % (окна по 2 мс)')
    ax.set_xlabel('мс от начала ответа (под профилировщиком)')
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('Загрузка GPU на протяжении всего ответа', fontweight='bold', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'gpu_busy.png')
    plt.close(fig)

    fig, (h, t) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [1, 1.3]})
    bins = np.logspace(0, 4, 41)
    for (kernels, _, _), label, color in ((eagle, 'EAGLE-3', COLORS['draft']), (plain, 'обычная', COLORS['verify'])):
        d = np.array([dur for _, dur, _, _ in kernels])
        h.hist(np.clip(d, 1, 1e4), bins=bins, histtype='step', lw=1.4, color=color,
               label=f'{label}: медиана {np.median(d):.0f} мкс, {100 * (d < 10).mean():.0f}% короче 10 мкс')
        stats[f'{label} median kernel us'] = float(np.median(d))
    h.set_xscale('log')
    h.set_xlabel('длительность ядра, мкс')
    h.set_ylabel('ядер')
    h.legend(frameon=False, fontsize=7.5)
    h.set_title('Большинство ядер — микросекунды', fontsize=9, loc='left')
    totals = {}
    for _, dur, name, _ in eagle[0]:
        totals[short(name)] = totals.get(short(name), 0) + dur
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:10][::-1]
    gpu_total = sum(totals.values())
    t.barh([n for n, _ in top], [100 * v / gpu_total for _, v in top], color=COLORS['draft'])
    t.set_xlabel('% всего GPU-времени (EAGLE-3)')
    t.tick_params(axis='y', labelsize=7)
    t.set_title('На что тратится время GPU', fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'kernels.png')
    plt.close(fig)

    (out / 'stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print('figures in', out, json.dumps(stats, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
