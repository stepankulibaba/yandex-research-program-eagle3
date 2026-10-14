"""The divergence of `ours` on 2026-10-12, for the report: training curves of the three runs so far.

    python code/make_incident_figures.py   (reads results/llama31-8b/nemo_incidents/<run>/{metrics,val}.jsonl,
                                            writes figures/incident_*.png)

Runs: ours with NeMo's recipe as published (pure bf16, lr to 2e-4; stopped by us at step 477), ours with fp32
master weights and the same lr (diverged at step ~580, lr 1.3e-4), and baseline (the authors' code) for reference.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IN, OUT = ROOT / 'results' / 'llama31-8b' / 'nemo_incidents', ROOT / 'figures'
RUNS = [  # (dir, label, color): fixed categorical slots 1-3 of the reference palette
    ('ours_fp32_lr2e-4_diverged', 'ours · NeMo, fp32 мастер-веса, lr→2e-4 (разошёлся)', '#2a78d6'),
    ('ours_bf16_stopped', 'ours · NeMo как в рецепте, чистый bf16 (остановлен)', '#eb6834'),
    ('baseline', 'baseline · код авторов', '#1baf7a'),
]


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                         'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    data = {d: (jsonl(IN / d / 'metrics.jsonl'), jsonl(IN / d / 'val.jsonl'))
            for d, _, _ in RUNS}
    xmax = max(m['samples'] for m in data['ours_fp32_lr2e-4_diverged'][0]) / 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle('Расхождение ours (NeMo, fp32 мастер-веса, lr 2e-4) · 12.10.2026', fontweight='bold')

    def panel(ax, key, title, runs, log=False):
        for d, label, color in RUNS:
            if d not in runs:
                continue
            rows = [m for m in data[d][0] if m.get(key) is not None and m['samples'] / 1e3 <= xmax]
            ys = [m[key][0] if isinstance(m[key], list) else m[key] for m in rows]
            xs = [m['samples'] / 1e3 for m in rows]
            if key in ('acc', 'lr'):     # every run starts at 0 (untrained, lr warmup from 0); the logged points are
                xs, ys = [0.0] + xs, [0.0] + ys       # window means, so the first one lies later
            ax.plot(xs, ys, color=color, lw=2, label=label)
        ax.set_title(title, loc='left')
        if log:
            ax.set_yscale('log')

    ours = ('ours_fp32_lr2e-4_diverged', 'ours_bf16_stopped')
    panel(axes[0, 0], 'ploss', 'Loss обучения ours (среднее 4 шагов TTT)', ours)
    panel(axes[0, 1], 'acc', 'Точность шага 1 на обучении', [d for d, _, _ in RUNS])
    panel(axes[1, 0], 'grad_norm', 'Норма градиента до клиппинга (лог. шкала)', ours, log=True)
    panel(axes[1, 1], 'lr', 'Learning rate', [d for d, _, _ in RUNS])
    spike = next(m for m in data['ours_fp32_lr2e-4_diverged'][0] if m['step'] >= 580)
    for ax in axes.flat:
        ax.axvline(spike['samples'] / 1e3, color='#52514e', ls='--', lw=1)
        ax.set_xlim(0, xmax)
    axes[0, 0].annotate(f'срыв: шаг 580, lr {spike["lr"]:.2g}', (spike['samples'] / 1e3, 6),
                        xytext=(8, 0), textcoords='offset points', color='#52514e', fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('диалогов, тыс.')
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = OUT / 'incident_ours_divergence.png'
    fig.savefig(out, dpi=130, facecolor='#fcfcfb')
    print(out)

    # validation tau estimate of the same runs (the authors' measure, 400 held-out dialogues)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d, label, color in RUNS:
        vals = [v for v in data[d][1] if v['samples'] / 1e3 <= xmax]
        ax.plot([v['samples'] / 1e3 for v in vals], [v['tau_estimate'] for v in vals], 'o-', color=color, lw=2,
                ms=6, label=label)
    ax.axvline(spike['samples'] / 1e3, color='#52514e', ls='--', lw=1)
    ax.set_xlabel('диалогов, тыс.')
    ax.set_title('τ-оценка на валидации', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    out = OUT / 'incident_ours_divergence_val.png'
    fig.savefig(out, dpi=130, facecolor='#fcfcfb')
    print(out)


if __name__ == '__main__':
    main()
