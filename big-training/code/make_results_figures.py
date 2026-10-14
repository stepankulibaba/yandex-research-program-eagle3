"""Everything we can show from the from-scratch run without the server: curves, gap, data efficiency, forecast.

    python code/make_results_figures.py    -> figures/*.png, results/llama31-8b/results.json, tables.md

Inputs (results/llama31-8b/, copied from the ClearML mirror and the final evaluation, part 1):
  <variant>/val.jsonl      validation tau estimate every 16k dialogues (400 held-out PerfectBlend dialogues)
  <variant>/metrics.jsonl  last 300 training log rows (every 50 steps)
  reference.json           the authors' official draft on the same validation set
  final_eval_part1.json    real tau in SGLang and in the authors' code (official, baseline)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / 'results' / 'llama31-8b', ROOT / 'figures'
COMPARE = 546.6                 # thousand dialogues, step 68,325 (0.4 epoch)
EPOCH = 1366.504                # thousand dialogues in the PerfectBlend training split
COLORS = {'baseline': '#1baf7a', 'ours': '#2a78d6', 'official': '#52514e'}
LABELS = {'baseline': 'baseline · слои 2/16/29', 'ours': 'ours · слои 16/22/29 + fc_norm',
          'official': 'официальный драфт авторов'}
BENCH = {'mt_bench': 'MT-bench', 'humaneval': 'HumanEval', 'gsm8k': 'GSM8K', 'alpaca': 'Alpaca', 'sum': 'CNN/DM'}


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def law(n, a, b, c):
    """Saturating power law in thousands of dialogues: tau -> a as n grows."""
    return a - b * n ** (-c)


def fit(n, t, start):
    keep = n >= start
    p, _ = curve_fit(law, n[keep], t[keep], p0=(4.2, 10, 0.5), maxfev=20000)
    resid = t[keep] - law(n[keep], *p)
    return p, float(resid.std(ddof=3))


def dialogues_for(tau, p):
    """Inverse of law(): thousands of dialogues to reach tau (inf if beyond the asymptote)."""
    a, b, c = p
    return float((b / (a - tau)) ** (1 / c)) if tau < a else float('inf')


def style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def main():
    FIG.mkdir(exist_ok=True)
    style()
    val = {v: jsonl(DATA / v / 'val.jsonl') for v in ('baseline', 'ours')}
    n = {v: np.array([x['samples'] / 1e3 for x in val[v]]) for v in val}
    t = {v: np.array([x['tau_estimate'] for x in val[v]]) for v in val}
    ref = json.loads((DATA / 'reference.json').read_text(encoding='utf-8'))
    real = json.loads((DATA / 'final_eval_part1.json').read_text(encoding='utf-8'))
    out = {'compare_k_dialogues': COMPARE, 'epoch_k_dialogues': EPOCH, 'official_val_tau': ref['tau_estimate']}

    # 1. gap at equal dialogues
    base_at = dict(zip(n['baseline'], t['baseline']))
    common = [(k, tau, base_at[k]) for k, tau in zip(n['ours'], t['ours']) if k in base_at and k > 0]
    ks = np.array([c[0] for c in common])
    gap = np.array([c[1] - c[2] for c in common])
    windows = [(0, 128), (128, 320), (320, 520)]
    out['gap'] = {
        'points': len(gap), 'ours_ahead': int((gap > 0).sum()),
        'mean': float(gap.mean()), 'last_5_mean': float(gap[-5:].mean()),
        'by_window': {f'{a}-{b}k': float(gap[(ks > a) & (ks <= b)].mean()) for a, b in windows},
        'last': {'k_dialogues': float(ks[-1]), 'ours': float(common[-1][1]), 'baseline': float(common[-1][2])},
        'table': [{'k_dialogues': float(k), 'ours': float(o), 'baseline': float(b), 'gap': float(o - b)}
                  for k, o, b in common]}

    # 2. fits and the noise of one validation point
    fits = {v: {s: fit(n[v], t[v], s) for s in (32, 64, 128)} for v in val}
    noise = float(np.mean([fits[v][64][1] for v in val]))
    out['val_noise_std'] = noise
    out['gap_in_noise_units'] = float(out['gap']['last_5_mean'] / noise)

    # 3. data efficiency: dialogues the baseline needs to reach ours' tau (fit from 64k, and plain interpolation)
    pb = fits['baseline'][64][0]
    eff = []
    for k, tau in zip(n['ours'], t['ours']):
        if k >= 64:
            nb = dialogues_for(tau, pb)
            raw = float(np.interp(tau, np.maximum.accumulate(t['baseline']), n['baseline'])) \
                if tau <= t['baseline'].max() else float('nan')
            eff.append({'k_dialogues': float(k), 'ours_tau': float(tau), 'baseline_needs_fit': nb,
                        'baseline_needs_interp': raw, 'ratio_fit': nb / k})
    out['data_efficiency'] = eff
    out['data_efficiency_mean_ratio_last_8'] = float(np.mean([e['ratio_fit'] for e in eff[-8:]]))

    # 4. forecast
    marks = {'compare_0.4_epoch': COMPARE, '1_epoch': EPOCH, '2_epochs': 2 * EPOCH, '3_epochs': 3 * EPOCH}
    fc = {}
    for v in val:
        fc[v] = {'params_from_64k': dict(zip('abc', map(float, fits[v][64][0]))),
                 'asymptote_range': [float(min(f[0][0] for f in fits[v].values())),
                                     float(max(f[0][0] for f in fits[v].values()))]}
        for name, k in marks.items():
            vals = [float(law(k, *f[0])) for f in fits[v].values()]
            fc[v][name] = {'mid': float(law(k, *fits[v][64][0])), 'low': min(vals), 'high': max(vals)}
    fc['ours_at_compare_by_gap'] = float(t['baseline'][-1] + out['gap']['last_5_mean'])
    # the asymptote itself is poorly pinned down by 0.4 epoch of data; what is robust is the gain per doubling
    fc['gain_per_doubling_measured'] = {v: {f'{a}->{2 * a}k': float(np.interp(2 * a, n[v], t[v]) - np.interp(a, n[v], t[v]))
                                            for a in (64, 128, 256)} for v in val}
    fc['gain_per_doubling_forecast'] = {v: {f'{a:.0f}->{2 * a:.0f}k': float(law(2 * a, *fits[v][64][0]) - law(a, *fits[v][64][0]))
                                            for a in (COMPARE, 2 * COMPARE, 4 * COMPARE)} for v in val}
    fc['k_dialogues_to_reach'] = {v: {str(x): {s: dialogues_for(x, f[0]) for s, f in fits[v].items()}
                                      for x in (3.9, 4.0, 4.1, 4.2)} for v in val}
    out['forecast'] = fc

    # 5. per-position acceptance at the last common point
    k_last = ks[-1]
    acc = {v: next(x['acc'] for x in val[v] if x['samples'] / 1e3 == k_last) for v in val}
    acc['official'] = ref['acc']
    out['val_acc_by_position'] = {'k_dialogues': float(k_last), **acc}

    # 6. training rows at equal steps (same data order in both runs, so a paired comparison)
    met = {v: {r['step']: r for r in jsonl(DATA / v / 'metrics.jsonl')} for v in val}
    steps = sorted(set(met['baseline']) & set(met['ours']))
    d_acc = np.array([met['ours'][s]['acc'][0] - met['baseline'][s]['acc'][0] for s in steps])
    d_loss = np.array([met['ours'][s]['ploss'][0] - met['baseline'][s]['ploss'][0] for s in steps])
    out['train_paired'] = {'steps': [steps[0], steps[-1]], 'rows': len(steps),
                           'acc0_diff_mean': float(d_acc.mean()), 'acc0_diff_se': float(d_acc.std() / len(steps) ** .5),
                           'acc0_ours_ahead_share': float((d_acc > 0).mean()),
                           'ploss0_diff_mean': float(d_loss.mean()), 'ploss0_diff_se': float(d_loss.std() / len(steps) ** .5)}

    # 7. real tau (part 1 of the final evaluation)
    s = real['sglang']
    out['real_tau'] = {d: {**s[d], 'mean': float(np.mean(list(s[d].values())))} for d in s}
    out['real_tau_authors_code'] = real['authors_code']
    out['real_tau_ours_projection'] = {
        'method': 'baseline real tau x (ours / baseline) on validation at the comparison point; not a measurement',
        'ratio': fc['ours_at_compare_by_gap'] / float(t['baseline'][-1]),
        **{b: float(s['baseline'][b] * fc['ours_at_compare_by_gap'] / t['baseline'][-1]) for b in s['baseline']}}

    (DATA / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    plots(n, t, fits, ref, ks, gap, noise, eff, acc, real, steps, met)
    tables(out, real)
    print(json.dumps({k: out[k] for k in ('gap', 'val_noise_std', 'gap_in_noise_units',
                                           'data_efficiency_mean_ratio_last_8', 'forecast', 'train_paired')},
                     indent=1, ensure_ascii=False)[:6000])


def plots(n, t, fits, ref, ks, gap, noise, eff, acc, real, steps, met):
    # validation curves + gap
    fig, (ax, dx) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for v in ('baseline', 'ours'):
        ax.plot(n[v], t[v], 'o-', color=COLORS[v], lw=2, ms=4, label=LABELS[v])
    ax.axhline(ref['tau_estimate'], color=COLORS['official'], ls='--', lw=1)
    ax.text(10, ref['tau_estimate'] + 0.03, f'официальный драфт: {ref["tau_estimate"]:.2f} (другие данные)',
            color=COLORS['official'], fontsize=8.5)
    ax.annotate('узел пропал\n(17:03, 526 тыс.)', xy=(n['ours'][-1], t['ours'][-1]), xytext=(380, 3.25),
                fontsize=8.5, color='#b23a48', arrowprops=dict(arrowstyle='->', color='#b23a48', lw=0.8))
    for a in (ax, dx):
        a.axvline(COMPARE, color='#b23a48', ls=':', lw=1)
    ax.set_ylim(2.0, 3.95)
    ax.set_xlim(0, 580)
    ax.set_ylabel('τ-оценка на валидации')
    ax.legend(loc='lower right', frameon=False)
    dx.bar(ks, gap, width=11, color=COLORS['ours'], alpha=0.85)
    dx.axhspan(-2 * noise, 2 * noise, color='#52514e', alpha=0.1, lw=0)
    dx.axhline(0, color='#52514e', lw=0.8)
    dx.text(575, -1.6 * noise, f'серая полоса: ±2σ шума одной точки ({2 * noise:.3f})', fontsize=8, color='#52514e', ha='right')
    dx.set_ylabel('ours − baseline')
    dx.set_xlabel('тыс. диалогов обучения')
    fig.savefig(FIG / 'val_tau.png')
    plt.close(fig)

    # forecast
    fig, ax = plt.subplots(figsize=(8, 4.4))
    grid = np.linspace(16, 3 * EPOCH, 400)
    for v in ('baseline', 'ours'):
        ax.plot(n[v], t[v], 'o', color=COLORS[v], ms=3.5, label=LABELS[v])
        curves = np.array([law(grid, *f[0]) for f in fits[v].values()])
        ax.plot(grid, law(grid, *fits[v][64][0]), color=COLORS[v], lw=1.6)
        ax.fill_between(grid, curves.min(0), curves.max(0), color=COLORS[v], alpha=0.15, lw=0)
    for k, name in ((COMPARE, '0.4 эпохи\n(сравнение)'), (EPOCH, '1 эпоха'), (2 * EPOCH, '2 эпохи'), (3 * EPOCH, '3 эпохи')):
        ax.axvline(k, color='#b23a48', ls=':', lw=0.9)
        ax.text(k + 25, 2.62, name, fontsize=8, color='#b23a48')
    ax.axhline(ref['tau_estimate'], color=COLORS['official'], ls='--', lw=1)
    ax.set_ylim(2.6, 4.6)
    ax.set_xlim(0, 3 * EPOCH + 150)
    ax.set_xlabel('тыс. диалогов обучения')
    ax.set_ylabel('τ-оценка на валидации')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.savefig(FIG / 'forecast.png')
    plt.close(fig)

    # data efficiency
    fig, ax = plt.subplots(figsize=(8, 3.4))
    k = [e['k_dialogues'] for e in eff]
    ax.plot(k, [100 * (e['ratio_fit'] - 1) for e in eff], 'o-', color=COLORS['ours'], lw=1.8, ms=4)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel('тыс. диалогов, которые увидел ours')
    ax.set_ylabel('baseline нужно больше, %')
    fig.savefig(FIG / 'data_efficiency.png')
    plt.close(fig)

    # per-position acceptance
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pos = np.arange(1, 8)
    for i, v in enumerate(('official', 'baseline', 'ours')):
        ax.bar(pos + (i - 1) * 0.27, acc[v], width=0.26, color=COLORS[v], label=LABELS[v])
    ax.set_ylim(0.6, 0.875)
    ax.set_xticks(pos)
    ax.set_xlabel('позиция в цепочке драфта (шаг TTT)')
    ax.set_ylabel('доля верных токенов')
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper right')
    fig.savefig(FIG / 'acc_by_position.png')
    plt.close(fig)

    # real tau
    fig, ax = plt.subplots(figsize=(8, 3.6))
    benches = list(BENCH)
    x = np.arange(len(benches))
    for i, d in enumerate(('official', 'baseline')):
        vals = [real['sglang'][d][b] for b in benches]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=COLORS[d], label=LABELS[d] + ' (SGLang)')
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ac = real['authors_code'][d]
        ax.plot([x[j] + (i - 0.5) * 0.36 for j, b in enumerate(benches) if b in ac], [ac[b] for b in benches if b in ac],
                'k_', ms=14, mew=1.6, label='код авторов' if i == 0 else None)
    ax.set_xticks(x, [BENCH[b] for b in benches])
    ax.set_ylim(0, 9.6)
    ax.set_ylabel('реальная τ (дерево 8/10/60)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=3)
    fig.savefig(FIG / 'real_tau.png')
    plt.close(fig)

    # paired training accuracy
    fig, ax = plt.subplots(figsize=(8, 3.2))
    d = np.array([met['ours'][s]['acc'][0] - met['baseline'][s]['acc'][0] for s in steps])
    ax.plot(steps, d, color=COLORS['ours'], lw=0.6, alpha=0.4)
    w = 20
    ax.plot(steps[w - 1:], np.convolve(d, np.ones(w) / w, 'valid'), color=COLORS['ours'], lw=2,
            label=f'скользящее среднее по {w} точкам')
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xlabel('шаг (8 диалогов на шаг; данные в одном порядке у обоих)')
    ax.set_ylabel('acc₀: ours − baseline')
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(FIG / 'train_paired.png')
    plt.close(fig)


def tables(out, real):
    g = out['gap']
    lines = ['# Таблицы', '', '## Валидация: τ-оценка при равном числе диалогов', '',
             '| тыс. диалогов | ours | baseline | ours − baseline |', '|---|---|---|---|']
    lines += [f"| {r['k_dialogues']:.0f} | {r['ours']:.3f} | {r['baseline']:.3f} | {r['gap']:+.3f} |" for r in g['table']]
    lines += ['', '## Реальная τ, SGLang (дерево 8/10/60, T = 0, 80 вопросов)', '',
              '| драфт | ' + ' | '.join(BENCH.values()) + ' | среднее |', '|---' * (len(BENCH) + 2) + '|']
    for d, r in out['real_tau'].items():
        lines.append(f'| {d} | ' + ' | '.join(f'{r[b]:.2f}' for b in BENCH) + f" | {r['mean']:.2f} |")
    p = out['real_tau_ours_projection']
    lines.append('| ours (прогноз, не замер) | ' + ' | '.join(f'≈{p[b]:.2f}' for b in BENCH) + ' | ' +
                 f"≈{np.mean([p[b] for b in BENCH]):.2f} |")
    lines += ['', '## Прогноз τ-оценки на валидации (степенной закон, разброс — по трём окнам подгонки)', '',
              '| | 0.4 эпохи | 1 эпоха | 2 эпохи | 3 эпохи | асимптота |', '|---|---|---|---|---|---|']
    for v in ('baseline', 'ours'):
        f = out['forecast'][v]
        cells = [f"{f[m]['mid']:.2f} ({f[m]['low']:.2f}–{f[m]['high']:.2f})"
                 for m in ('compare_0.4_epoch', '1_epoch', '2_epochs', '3_epochs')]
        lines.append(f'| {v} | ' + ' | '.join(cells) + f" | {f['asymptote_range'][0]:.2f}–{f['asymptote_range'][1]:.2f} |")
    (DATA / 'tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
