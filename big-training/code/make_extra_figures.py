"""Extra figures for slides and the full report (the core ones are in make_results_figures.py).

    python code/make_extra_figures.py      -> figures/x_*.png, results/llama31-8b/extra.json

Reads only results/llama31-8b/ (collect_results.py). Figures whose data is missing are skipped.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / 'results' / 'llama31-8b', ROOT / 'figures'
EV = DATA / 'final_eval'
COLORS = {'baseline': '#1baf7a', 'ours': '#2a78d6', 'official': '#52514e', 'bug': '#e34948', 'muted': '#9a9893'}
LABELS = {'baseline': 'baseline · слои 2/16/29', 'ours': 'ours · слои 16/22/29 + fc_norm',
          'official': 'официальный драфт авторов'}
BENCH = {'mt_bench': 'MT-bench', 'humaneval': 'HumanEval', 'gsm8k': 'GSM8K', 'alpaca': 'Alpaca', 'sum': 'CNN/DM'}
SLOT_COLORS = ['#f2b134', '#8a63d2', '#e8590c']
COMPARE_STEP = 68325
DIALOGUES_PER_STEP = 8


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def save(fig, name):
    fig.savefig(FIG / f'x_{name}.png')
    plt.close(fig)
    print('figure', name)


def tau_of(rows):
    return sum(r['tokens'] - r['turns'] for r in rows) / sum(r['rounds'] for r in rows)


def answers(draft, bench):
    p = EV / draft / f'sglang_{bench}.json'
    return read(p) if p.exists() else None


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, 'valid')


# ---------------------------------------------------------------- what the draft takes from each layer
def contributions(extra):
    p = DATA / 'milestones' / 'contributions.json'
    if not p.exists():
        return
    c = read(p)
    final = {v: c['variants'][v][-1] for v in ('baseline', 'ours')}
    extra['contributions_final'] = {v: {'layers': final[v]['layers'], 'share': final[v]['contrib_share']}
                                    for v in final}

    fig, ax = plt.subplots(figsize=(8, 2.6))
    for y, v in enumerate(('ours', 'baseline')):
        left = 0
        for i, (layer, share) in enumerate(zip(final[v]['layers'], final[v]['contrib_share'])):
            ax.barh(y, share, left=left, color=SLOT_COLORS[i], height=0.6, edgecolor='white')
            if share > 0.04:
                ax.text(left + share / 2, y, f'слой {layer}\n{100 * share:.0f} %', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
            else:
                ax.annotate(f'слой {layer}: {100 * share:.1f} %', xy=(left + share / 2, y + 0.3),
                            xytext=(0.02, y + 0.48), fontsize=8.5, color='#52514e',
                            arrowprops=dict(arrowstyle='-', color='#52514e', lw=0.6))
            left += share
    ax.set_yticks([0, 1], [LABELS['ours'], LABELS['baseline']])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f'{int(x * 100)} %' for x in np.linspace(0, 1, 6)])
    ax.grid(False)
    ax.set_xlabel('доля выхода fc от каждого слоя цели (64 диалога валидации, конец обучения)')
    save(fig, 'layer_contribution')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, v in zip(axes, ('baseline', 'ours')):
        rows = c['variants'][v]
        k = [r['step'] * DIALOGUES_PER_STEP / 1e3 for r in rows]
        for i, layer in enumerate(rows[0]['layers']):
            ax.plot(k, [100 * r['contrib_share'][i] for r in rows], 'o-', color=SLOT_COLORS[i], lw=2, ms=4,
                    label=f'слой {layer}')
        ax.set_title(LABELS[v], fontsize=10, color=COLORS[v])
        ax.set_xlabel('тыс. диалогов обучения')
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel('доля выхода fc, %')
    save(fig, 'layer_contribution_training')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    layers = sorted(int(k) for k in c['state_norm'])
    axes[0].bar([str(x) for x in layers], [c['state_norm'][str(x)] for x in layers], color=COLORS['muted'])
    for x, l in enumerate(layers):
        axes[0].text(x, c['state_norm'][str(l)], f"{c['state_norm'][str(l)]:.0f}", ha='center', va='bottom', fontsize=8.5)
    axes[0].set_xlabel('слой цели (вход слоя)')
    axes[0].set_ylabel('средняя норма состояния')
    axes[0].set_title('масштаб признаков разных слоёв', fontsize=10)
    rows = c['variants']['ours']
    k = [r['step'] * DIALOGUES_PER_STEP / 1e3 for r in rows]
    for i, layer in enumerate(rows[0]['layers']):
        axes[1].plot(k, [r['fc_norm_scale'][i]['mean'] for r in rows], 'o-', color=SLOT_COLORS[i], lw=2, ms=4,
                     label=f'слой {layer}')
    axes[1].set_xlabel('тыс. диалогов обучения')
    axes[1].set_ylabel('средний масштаб fc_norm')
    axes[1].set_title('ours: обучаемые масштабы fc_norm', fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5)
    save(fig, 'state_norms_fc_norm')


# ---------------------------------------------------------------- acceptance in real generation
def acceptance(extra):
    benches = [b for b in BENCH if answers('ours', b)]
    if not benches:
        return
    fig, axes = plt.subplots(1, len(benches), figsize=(2.6 * len(benches), 3.2), sharey=True)
    extra['accept_mean_tokens'] = {}
    for ax, b in zip(axes, benches):
        for i, d in enumerate(('official', 'baseline', 'ours')):
            a = answers(d, b)
            h = np.sum([r['accept_hist'] for r in a['per_question']], axis=0).astype(float)
            h /= h.sum()
            ax.bar(np.arange(9) + (i - 1) * 0.28, 100 * h, width=0.27, color=COLORS[d], label=LABELS[d])
            extra['accept_mean_tokens'].setdefault(d, {})[b] = float((np.arange(9) * h).sum())
        ax.set_title(BENCH[b], fontsize=10)
        ax.set_xticks(range(0, 9, 2))
        ax.set_xlabel('принято токенов драфта')
    axes[0].set_ylabel('доля раундов проверки, %')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.13))
    save(fig, 'accept_histogram')

    # where ours gains: share of rounds with >= k accepted tokens, ours minus baseline
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for b in benches:
        cdf = {}
        for d in ('baseline', 'ours'):
            h = np.sum([r['accept_hist'] for r in answers(d, b)['per_question']], axis=0).astype(float)
            h /= h.sum()
            cdf[d] = np.array([h[k:].sum() for k in range(1, 9)])
        ax.plot(range(1, 9), 100 * (cdf['ours'] - cdf['baseline']), 'o-', lw=1.8, ms=4, label=BENCH[b])
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel('принято не меньше k токенов драфта за раунд')
    ax.set_ylabel('ours − baseline, п. п.')
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    save(fig, 'accept_tail_gain')


def per_question(extra):
    benches = [b for b in BENCH if answers('ours', b)]
    if not benches:
        return
    fig, axes = plt.subplots(1, len(benches), figsize=(2.6 * len(benches), 2.9))
    extra['per_question_ours_better_share'] = {}
    for ax, b in zip(axes, benches):
        qa = {r['question_id']: r for r in answers('ours', b)['per_question']}
        qb = {r['question_id']: r for r in answers('baseline', b)['per_question']}
        ids = sorted(set(qa) & set(qb))
        x = np.array([tau_of([qb[i]]) for i in ids])
        y = np.array([tau_of([qa[i]]) for i in ids])
        ax.scatter(x, y, s=10, color=COLORS['ours'], alpha=0.6, lw=0)
        lo, hi = min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], color='#52514e', lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        better, worse = float((y > x + 1e-9).mean()), float((y < x - 1e-9).mean())
        extra['per_question_ours_better_share'][b] = {'better': better, 'worse': worse, 'equal': 1 - better - worse}
        ax.set_title(f'{BENCH[b]}\nours выше: {100 * better:.0f} %, ниже: {100 * worse:.0f} %', fontsize=8.5)
        ax.set_xlabel('τ baseline', fontsize=9)
    axes[0].set_ylabel('τ ours')
    save(fig, 'per_question_scatter')


def speed_vs_tau(extra):
    pts = []
    for d in sorted(p.name for p in EV.iterdir() if p.is_dir() and p.name != 'plain'):
        for b in BENCH:
            a = answers(d, b)
            # milestone runs carry gpu_shared=True only because milestone_analysis.py's own process kept a CUDA
            # context (5 GB, no compute) while SGLang ran; no other workload shared the GPU
            if a and (not a['gpu_shared'] or '_step' in d) and d != 'ours_sglang_capture_bug':
                pts.append((d, b, tau_of(a['per_question']), a['tok_s']))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = dict(zip(BENCH, 'osD^v'))
    for d, b, t, s in pts:
        kind = 'official' if d == 'official' else ('ours' if d.startswith('ours') else 'baseline')
        final = d in ('official', 'baseline', 'ours')
        ax.scatter(t, s, marker=markers[b], s=55 if final else 22, color=COLORS[kind], alpha=1 if final else 0.45,
                   edgecolor='white', lw=0.5)
    for b, m in markers.items():
        ax.scatter([], [], marker=m, color='#52514e', label=BENCH[b])
    for k in ('official', 'baseline', 'ours'):
        ax.scatter([], [], marker='s', color=COLORS[k], label=LABELS[k])
    t = np.array([p[2] for p in pts])
    s = np.array([p[3] for p in pts])
    extra['speed_tau_corr'] = float(np.corrcoef(t, s)[0, 1])
    ax.set_xlabel('реальная τ')
    ax.set_ylabel('токенов в секунду (SGLang, batch 1)')
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left')
    ax.text(0.98, 0.04, f'корреляция {extra["speed_tau_corr"]:.2f}; мелкие точки — промежуточные чекпоинты',
            transform=ax.transAxes, ha='right', fontsize=8, color='#52514e')
    save(fig, 'speed_vs_tau')


def sglang_bug(extra):
    bug = {b: answers('ours_sglang_capture_bug', b) for b in BENCH}
    if not all(bug.values()):
        return
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(BENCH))
    bars = [('без исправления', 'ours_sglang_capture_bug', COLORS['bug']), ('с исправлением', 'ours', COLORS['ours'])]
    for i, (label, d, color) in enumerate(bars):
        vals = [tau_of(answers(d, b)['per_question']) for b in BENCH]
        r = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=color, label=f'ours в SGLang, {label}')
        ax.bar_label(r, fmt='%.2f', fontsize=8, padding=2)
    for j, b in enumerate(BENCH):
        p = EV / 'ours' / f'authors_{b}.json'
        if p.exists():
            ax.plot(x[j] + 0.18, read(p)['tau'], 'k_', ms=14, mew=1.6, label='код авторов' if j == 0 else None)
    ax.set_xticks(x, list(BENCH.values()))
    ax.set_ylim(0, 9.3)
    ax.set_ylabel('реальная τ')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=3)
    save(fig, 'sglang_bug')


# ---------------------------------------------------------------- real tau at the milestones
def milestones(extra):
    rows = {}
    for d in EV.iterdir():
        if '_step' in d.name:
            v, step = d.name.split('_step')
            for b in BENCH:
                a = answers(d.name, b)
                if a:
                    rows.setdefault(v, {}).setdefault(b, {})[int(step)] = tau_of(a['per_question'])
    if not rows:
        return
    for v in ('baseline', 'ours'):
        for b in BENCH:
            a = answers(v, b)
            if a and b in rows.get(v, {}):
                rows[v][b][COMPARE_STEP] = tau_of(a['per_question'])
    extra['real_tau_milestones'] = {v: {b: {str(k): t for k, t in sorted(d.items())} for b, d in bs.items()}
                                    for v, bs in rows.items()}
    benches = [b for b in BENCH if b in rows.get('ours', {})]
    fig, axes = plt.subplots(1, len(benches), figsize=(2.7 * len(benches), 3.2))
    for ax, b in zip(np.atleast_1d(axes), benches):
        for v in ('baseline', 'ours'):
            d = rows.get(v, {}).get(b, {})
            ks = sorted(d)
            ax.plot([k * DIALOGUES_PER_STEP / 1e3 for k in ks], [d[k] for k in ks], 'o-', color=COLORS[v], lw=2, ms=4,
                    label=LABELS[v])
        off = answers('official', b)
        if off:
            ax.axhline(tau_of(off['per_question']), color=COLORS['official'], ls='--', lw=1)
        ax.set_title(BENCH[b], fontsize=10)
        ax.set_xlabel('тыс. диалогов')
    np.atleast_1d(axes)[0].set_ylabel('реальная τ (SGLang)')
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color=COLORS['official'], ls='--')], labels + [LABELS['official']],
               frameon=False, fontsize=8.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.15))
    save(fig, 'real_tau_milestones')

    # the validation proxy against real tau
    val = {v: {r['step']: r['tau_estimate'] for r in jsonl(DATA / v / 'val.jsonl')} for v in ('baseline', 'ours')}
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys = [], []
    for v in ('baseline', 'ours'):
        for b in benches:
            d = rows.get(v, {}).get(b, {})
            ks = [k for k in sorted(d) if k in val[v]]
            ax.plot([val[v][k] for k in ks], [d[k] for k in ks], 'o-', color=COLORS[v], lw=1, ms=4, alpha=0.8)
            xs += [val[v][k] for k in ks]
            ys += [d[k] for k in ks]
    for v in ('baseline', 'ours'):
        ax.plot([], [], 'o-', color=COLORS[v], label=LABELS[v])
    ax.set_xlabel('τ-оценка на валидации (одна цепочка)')
    ax.set_ylabel('реальная τ в SGLang (дерево)')
    ax.legend(frameon=False, fontsize=8.5, loc='center right')
    for b in benches:
        d = rows['ours'][b]
        k = max(d)
        ax.text(val['ours'][k] + 0.01, d[k], BENCH[b], fontsize=8, va='center', color='#52514e')
    corr = {}
    for b in benches:          # within a benchmark, over the milestones of both variants
        pairs = [(val[v][k], rows[v][b][k]) for v in ('baseline', 'ours') for k in rows[v][b] if k in val[v]]
        corr[b] = float(np.corrcoef(*zip(*pairs))[0, 1])
    extra['val_vs_real_corr'] = corr
    ax.text(0.98, 0.03, 'корреляция внутри бенчмарка:\n' + ', '.join(f'{BENCH[b]} {c:.3f}' for b, c in corr.items()),
            transform=ax.transAxes, fontsize=7.5, ha='right', color='#52514e')
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12)
    save(fig, 'val_vs_real')

    # ours - baseline at each milestone
    fig, ax = plt.subplots(figsize=(8, 3.4))
    extra['real_tau_milestones_delta'] = {}
    for b, color in zip(benches, ('#2a78d6', '#f2b134', '#8a63d2', '#e34948')):
        ks = sorted(set(rows['ours'][b]) & set(rows['baseline'][b]))
        d = [rows['ours'][b][k] - rows['baseline'][b][k] for k in ks]
        extra['real_tau_milestones_delta'][b] = {str(k): x for k, x in zip(ks, d)}
        ax.plot([k * DIALOGUES_PER_STEP / 1e3 for k in ks], d, 'o-', color=color, lw=1.8, ms=4,
                label=f'{BENCH[b]}: ours впереди в {sum(x > 0 for x in d)} из {len(d)}')
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel('тыс. диалогов обучения')
    ax.set_ylabel('реальная τ: ours − baseline')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, 'real_tau_milestones_delta')


# ---------------------------------------------------------------- training
def training(extra):
    met = {v: jsonl(DATA / v / 'metrics.jsonl') for v in ('baseline', 'ours')}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    w = 20
    for v, rows in met.items():
        k = np.array([r['samples'] / 1e3 for r in rows])
        acc0 = [r['acc'][0] for r in rows]
        acc6 = [r['acc'][-1] for r in rows]
        loss = [r['ploss'][0] for r in rows]
        axes[0, 0].plot(k[w - 1:], smooth(acc0, w), color=COLORS[v], lw=1.6, label=LABELS[v])
        axes[0, 1].plot(k[w - 1:], smooth(acc6, w), color=COLORS[v], lw=1.6)
        axes[1, 0].plot(k[w - 1:], smooth(loss, w), color=COLORS[v], lw=1.6)
        axes[1, 1].plot(k, [r['lr'] * 1e5 for r in rows], color=COLORS[v], lw=1.6)
    axes[0, 0].set_ylabel('точность 1-го токена (обучение)')
    axes[0, 1].set_ylabel('точность 7-го токена (обучение)')
    axes[1, 0].set_ylabel('лосс 1-й позиции (обучение)')
    axes[1, 1].set_ylabel('lr, ×1e-5')
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel('тыс. диалогов')
    axes[1, 1].text(0.98, 0.05, 'разогрев, затем линейный спад до 0 к концу эпохи;\nостановились на 0,4 эпохи',
                    transform=axes[1, 1].transAxes, ha='right', fontsize=8, color='#52514e')
    save(fig, 'training_curves')

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
    for v, rows in met.items():
        k = [r['samples'] / 1e3 for r in rows]
        axes[0].plot(k, [r['loss_scale'] for r in rows], color=COLORS[v], lw=1.4, label=LABELS[v])
        axes[1].plot(k, [r['skipped_steps'] for r in rows], color=COLORS[v], lw=1.4)
    axes[0].set_yscale('log', base=2)
    axes[0].set_ylabel('fp16 loss scale')
    axes[1].set_ylabel('пропущено шагов\n(переполнение fp16)')
    axes[1].set_xlabel('тыс. диалогов')
    axes[0].legend(frameon=False, fontsize=8.5)
    extra['skipped_steps'] = {v: rows[-1]['skipped_steps'] for v, rows in met.items()}
    save(fig, 'fp16_stability')

    # wall clock: sessions on one H200
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for v, rows in met.items():
        t = [datetime.fromtimestamp(r['time']) for r in rows]
        ax.plot(t, [r['samples'] / 1e3 for r in rows], '.', ms=2.5, color=COLORS[v], label=LABELS[v])
    ax.axvspan(datetime(2026, 10, 12, 16, 0), datetime(2026, 10, 13, 0, 12), color=COLORS['bug'], alpha=0.08, lw=0)
    ax.text(datetime(2026, 10, 12, 20, 0), 300, 'ours:\nпопытки\nна NeMo', ha='center', fontsize=8,
            color=COLORS['bug'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    ax.set_ylabel('тыс. диалогов')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', markerscale=4)
    ax.text(0.99, 0.04, 'разрывы — смена 8-часовых сессий SLURM; вариант продолжает с того же диалога',
            transform=ax.transAxes, ha='right', fontsize=8, color='#52514e')
    fig.autofmt_xdate()
    save(fig, 'timeline')

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for v, rows in met.items():
        t = [datetime.fromtimestamp(r['time']) for r in rows]
        ax.plot(t, [r['samples_per_s'] for r in rows], '.', ms=2.5, color=COLORS[v], label=LABELS[v])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    ax.set_ylabel('диалогов в секунду')
    ax.legend(frameon=False, fontsize=8.5, markerscale=4)
    ax.text(0.99, 0.9, 'вместе ≈ 4,4 на вариант, один ≈ 9,7', transform=ax.transAxes, ha='right', fontsize=8,
            color='#52514e')
    fig.autofmt_xdate()
    save(fig, 'throughput')


def validation_positions(extra):
    val = {v: jsonl(DATA / v / 'val.jsonl') for v in ('baseline', 'ours')}
    by = {v: {r['samples']: r for r in rows} for v, rows in val.items()}
    common = sorted(k for k in set(by['baseline']) & set(by['ours']) if k > 0)
    gap = np.array([[by['ours'][k]['acc'][p] - by['baseline'][k]['acc'][p] for k in common] for p in range(7)])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(100 * gap, aspect='auto', cmap='RdBu', vmin=-1.2, vmax=1.2, origin='lower',
                   extent=[common[0] / 1e3 - 8, common[-1] / 1e3 + 8, 0.5, 7.5])
    ax.set_yticks(range(1, 8))
    ax.set_ylabel('позиция в цепочке')
    ax.set_xlabel('тыс. диалогов')
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='ours − baseline, п. п.')
    extra['val_acc_gap_share_positive'] = float((gap > 0).mean())
    save(fig, 'val_position_gap')

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for v, rows in val.items():
        k = [r['samples'] / 1e3 for r in rows if r['samples'] > 0]
        for p in (0, 3, 6):
            ax.plot(k, [r['acc'][p] for r in rows if r['samples'] > 0], color=COLORS[v], lw=1.6,
                    ls=['-', '--', ':'][p // 3], label=f'{v}, позиция {p + 1}')
    ax.set_xlabel('тыс. диалогов')
    ax.set_ylabel('точность на валидации')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, 'val_positions')


def data_overview(extra):
    p = DATA / 'data_stats.json'
    if not p.exists():
        return
    st = read(p)
    h = st['length_histogram_512']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={'width_ratios': [1.3, 1]})
    axes[0].bar(list(h), [x / 1e3 for x in h.values()], color=COLORS['muted'])
    axes[0].set_xlabel('длина диалога, токенов')
    axes[0].set_ylabel('тыс. диалогов')
    axes[0].set_title('PerfectBlend-Regenerated (LLaMA-3.1-8B-Instruct)', fontsize=10)
    used = 546600
    axes[1].barh(['обучение', 'увидели до\nточки сравнения'], [st['train'] / 1e6, used / 1e6],
                 color=[COLORS['muted'], COLORS['ours']])
    axes[1].set_xlabel('млн диалогов')
    for y, x in enumerate([st['train'] / 1e6, used / 1e6]):
        axes[1].text(x, y, f' {x:.2f}', va='center', fontsize=9)
    axes[1].set_xlim(0, 1.7)
    axes[1].set_title(f"{st['train_tokens'] / 1e9:.2f} млрд токенов; словарь драфта 32 тыс. покрывает "
                      f"{100 * st['draft_vocab_coverage']:.1f} %", fontsize=8.5)
    save(fig, 'data')


def main():
    FIG.mkdir(exist_ok=True)
    style()
    extra = {}
    for f in (contributions, acceptance, per_question, speed_vs_tau, sglang_bug, milestones, training,
              validation_positions, data_overview):
        f(extra)
    (DATA / 'extra.json').write_text(json.dumps(extra, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(extra, indent=1, ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
