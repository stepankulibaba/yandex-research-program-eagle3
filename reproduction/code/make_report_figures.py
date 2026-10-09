"""Figures of the reproduction report from results/numbers.json -> figures/*.png.

    python code/make_report_figures.py        (from the reproduction/ folder or anywhere)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / 'results/numbers.json').read_text(encoding='utf-8'))
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

PAPER, OURS, OURS_OLD, NEMO, AUTHORS = '#9AA3AD', '#2F6DB5', '#9CC0E8', '#2E7D4F', '#C2703D'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.dpi': 150})
BENCHES = DATA['benches']


def bars(ax, groups, series, colors, labels, fmt='{:.2f}'):
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for i, (values, color, label) in enumerate(zip(series, colors, labels)):
        positions = x + (i - (len(series) - 1) / 2) * width
        ax.bar(positions, values, width, color=color, label=label)
        for p, v in zip(positions, values):
            ax.text(p, v, fmt.format(v), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x, groups)


def fig_3a():
    """Speed-up and tau per benchmark: paper vs the authors' code on our H200 (depth 8)."""
    paper, ours = DATA['task_3a']['paper'], DATA['task_3a']['ours_depth8']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for row, temp in enumerate(('T0', 'T1')):
        title_t = 'T = 0' if temp == 'T0' else 'T = 1'
        for col, key, name in ((0, 'tau', 'τ (токенов за проход большой модели)'), (1, 'speedup', 'ускорение, ×')):
            ax = axes[row, col]
            bars(ax, BENCHES, [paper[temp][key], ours[temp][key]], [PAPER, OURS], ['статья', 'код авторов на H200'])
            ax.set_title(f'{name}, {title_t}', fontsize=10)
            ax.set_ylim(0, max(paper[temp][key]) * 1.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle('3a. τ совпадает со статьёй, ускорение ниже', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / 'fig1_3a_tau_speedup.png')
    plt.close(fig)


def fig_depth():
    """tau at the scripts' default depth (EAGLE-2 tree) vs the paper's depth 8, T=0."""
    d = DATA['task_3a']
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars(ax, BENCHES, [d['ours_depth6_default']['T0']['tau'], d['ours_depth8']['T0']['tau'], d['paper']['T0']['tau']],
         [OURS_OLD, OURS, PAPER], ['глубина 6 (по умолчанию в скрипте авторов)', 'глубина 8 (как в статье)', 'статья'])
    ax.set_ylim(0, 7.8)
    ax.set_ylabel('τ, T = 0')
    ax.legend(loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig2_3a_depth.png')
    plt.close(fig)


def fig_cycle():
    """One plain step vs one EAGLE round, ms, with the paper's ratio for reference."""
    p = DATA['profile_3a']
    plain = p['plain_step_ms']['target forward'] + p['plain_step_ms']['loop']
    r = p['eagle_round_ms']
    rest = r['accept'] + r['update'] + r['loop'] + r['prefill']
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.barh(1, p['plain_step_ms']['target forward'], color=AUTHORS, label='прогон большой модели')
    ax.barh(1, p['plain_step_ms']['loop'], left=p['plain_step_ms']['target forward'], color='#DDDDDD')
    ax.barh(0, r['verify'], color=AUTHORS)
    ax.barh(0, r['draft'], left=r['verify'], color=OURS, label='драфт: 8 шагов дерева')
    ax.barh(0, rest, left=r['verify'] + r['draft'], color='#DDDDDD', label='выбор ветки, обновление, Python')
    total = r['verify'] + r['draft'] + rest
    ax.text(plain + 0.3, 1, f'{plain:.1f} мс', va='center')
    ax.text(total + 0.3, 0, f'{total:.1f} мс = {total / plain:.2f} обычного шага', va='center')
    ax.axvline(1.4 * plain, color=PAPER, ls='--', lw=1)
    ax.text(1.4 * plain, 1.45, 'цикл у авторов ≈ 1.4 шага', color='#666666', fontsize=8, ha='center')
    ax.set_yticks([0, 1], ['цикл EAGLE', 'обычный шаг'])
    ax.set_xlim(0, 34)
    ax.set_xlabel('мс на H200 (код авторов, MT-bench)')
    ax.legend(loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig3_cycle_profile.png')
    plt.close(fig)


def fig_training():
    """Training throughput: the authors' trainer as published vs NeMo; and on identical (padded) work."""
    runs = DATA['task_4']['runs']
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [2.2, 1]})
    names = ['original_asis', 'nemo_eager_pack', 'nemo_compile', 'nemo_compile_fp8']
    values = [runs[n]['tok_s'] for n in names]
    labels = ['авторы\nкак есть', 'NeMo\npacking', 'NeMo\n+ compile', 'NeMo\n+ compile + FP8']
    a.bar(labels, values, color=[AUTHORS, NEMO, NEMO, NEMO])
    for i, v in enumerate(values):
        a.text(i, v, f'{int(v + 0.5):,}\n×{v / values[0]:.2f}'.replace(',', ' '), ha='center', va='bottom', fontsize=8)
    a.set_ylim(0, max(values) * 1.25)
    a.set_ylabel('токенов в секунду')
    a.set_title('Каждый в своих настройках', fontsize=10)
    m = [runs['original']['tok_s'], runs['nemo_eager_nopack']['tok_s']]
    b.bar(['авторы', 'NeMo'], m, color=[AUTHORS, NEMO])
    for i, v in enumerate(m):
        b.text(i, v, f'{int(v + 0.5):,}\n×{v / m[0]:.2f}'.replace(',', ' '), ha='center', va='bottom', fontsize=8)
    b.set_ylim(0, max(m) * 1.3)
    b.set_title('Одинаковая работа (паддинг до 2048)', fontsize=10)
    fig.suptitle('4. Скорость шага обучения драфта, одна H200', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig4_training_speed.png')
    plt.close(fig)


if __name__ == '__main__':
    fig_3a()
    fig_depth()
    fig_cycle()
    fig_training()
    print('figures written to', FIGURES)
