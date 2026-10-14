"""Gradient norm per draft weight on a healthy and a collapsed NeMo checkpoint (train_nemo_full.py --diagnose,
2026-10-12 23:16, the same 32 packed rows), for the report.

    python code/make_diag_figure.py        -> figures/diag_grads.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / 'figures'
# weight: (median grad norm at step 790, median at step 933) — from the diagnosis log
DATA = [('q_proj', 1.01, 24.9), ('k_proj', 0.966, 25.9), ('fc', 1.55, 22.0), ('v_proj', 1.23, 8.11),
        ('MLP down_proj', 1.40, 7.30), ('MLP gate_proj', 1.16, 7.05), ('o_proj', 1.16, 6.43), ('lm_head', 2.12, 2.42)]


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#52514e', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e'})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [d[0] for d in DATA][::-1]
    ys = range(len(names))
    healthy = [d[1] for d in DATA][::-1]
    broken = [d[2] for d in DATA][::-1]
    ax.barh([y + 0.2 for y in ys], healthy, height=0.38, color='#1baf7a', label='шаг 790 — здоровая модель')
    ax.barh([y - 0.2 for y in ys], broken, height=0.38, color='#e34948', label='шаг 933 — внутри срыва')
    for y, h, b in zip(ys, healthy, broken):
        ax.text(b * 1.08, y - 0.2, f'×{b / h:.0f}' if b / h >= 2 else f'×{b / h:.1f}', va='center', fontsize=9,
                color='#52514e')
    ax.set_yticks(list(ys), names)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 60)
    ax.set_xlabel('медианная норма градиента на одной упакованной строке (лог. шкала)')
    ax.grid(axis='x', color='#e6e5e1', linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    out = OUT / 'diag_grads.png'
    fig.savefig(out, dpi=150, facecolor='white')
    print(out)


if __name__ == '__main__':
    main()
