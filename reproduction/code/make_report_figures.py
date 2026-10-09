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


def fig_setups():
    """T=0 speed-up per benchmark: the paper, the authors' code on A100 and H200, SGLang on H200 (paper tree)."""
    d = DATA['task_3a']
    series = [d['paper']['T0']['speedup'], d['a100_depth8']['T0']['speedup'], d['ours_depth8']['T0']['speedup'],
              DATA['task_3b']['T0']['paper_8_10_60']['speedup']]
    labels = ['статья', 'код авторов, A100', 'код авторов, H200', 'SGLang, H200']
    labels = [f'{name} (ср. {np.mean(s):.2f}×)' for name, s in zip(labels, series)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bars(ax, BENCHES, series, [PAPER, AUTHORS, OURS, NEMO], labels)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel('ускорение, T = 0')
    ax.legend(loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig8_speedup_setups.png')
    plt.close(fig)


TREE_LABELS = {'paper_8_10_60': '8 / 10 / 60\n(как в статье)', 'eagle2_6_10_60': '6 / 10 / 60\n(EAGLE-2)',
               'docs_5_8_32': '5 / 8 / 32\n(доки SGLang)', 'chain_3_1_4': '3 / 1 / 4\n(NeMo по умолчанию)'}


def fig_trees():
    """3b: mean tau and speed-up over the benchmarks for each SGLang tree (steps / top-k / tokens), T=0."""
    t0 = DATA['task_3b']['T0']
    names = list(TREE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title, ref in ((axes[0], 'tau', 'τ', np.mean(DATA['task_3a']['paper']['T0']['tau'])),
                                (axes[1], 'speedup', 'ускорение, ×', np.mean(DATA['task_3a']['ours_depth8']['T0']['speedup']))):
        values = [np.mean(t0[n][key]) for n in names]
        ax.bar([TREE_LABELS[n] for n in names], values, color=[NEMO, OURS_OLD, OURS_OLD, '#C9CED4'])
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.axhline(ref, color=AUTHORS if key == 'speedup' else PAPER, ls='--', lw=1)
        ax.text(3.45, ref, 'код авторов, H200' if key == 'speedup' else 'статья', ha='right', va='bottom',
                fontsize=8, color='#555555')
        ax.set_title(f'{title}, среднее по 5 бенчмаркам', fontsize=10)
        ax.set_ylim(0, max(values + [ref]) * 1.2)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('3b. SGLang на H200, T = 0: шаги драфта / top-k / токенов в дереве', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig9_3b_trees.png')
    plt.close(fig)


def fig_budget():
    """5: GPU-hours for the full corpus versus epochs, the 100 h budget as a line."""
    b = DATA['task_5']
    epochs = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, label, color, dy in (('original_asis', 'код авторов как есть', AUTHORS, 8),
                                   ('nemo_compile_fp8', 'NeMo + compile + FP8', NEMO, -16)):
        total = b['reserve_h'] + b['regen_h'] + epochs * b['h_per_epoch'][name]
        forty = b['total_h_40_epochs'][name]
        ax.plot(epochs, total, 'o-', color=color,
                label=f"{label}: {b['h_per_epoch'][name]:.1f} ч/эпоха; 40 эпох = {forty:.0f} ч")
        for e, t in zip(epochs, total):
            if e in (2, 3, 5, 10):
                ax.text(e, t + dy, f'{t:.0f}', ha='center', fontsize=7, color=color)
    ax.axhline(b['budget_h'], color='#B23A48', lw=1.2)
    ax.text(9.9, b['budget_h'] - 14, 'бюджет 100 ч', color='#B23A48', fontsize=8, ha='right')
    ax.set_xticks(epochs)
    ax.set_ylim(0, 280)
    ax.set_xlabel('эпох по всем данным (~532 тыс. диалогов)')
    ax.set_ylabel('GPU-часов H200')
    ax.set_title(f"5. Регенерация {b['regen_h']:.0f} ч + эпохи + {b['reserve_h']} ч запаса", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig10_budget.png')
    plt.close(fig)


def fig_regimes():
    """Cost of an EAGLE round in plain steps on H200 and A100: by GPU work, by kernel launches, and measured."""
    h, a = DATA['profile_3a'], DATA['profile_a100']
    def ratio(phases):      # all phases of a round over one plain step
        return sum(v for k, v in phases.items() if k != 'plain') / phases['plain']
    h_gpu, h_launch = ratio(h['phase_gpu_ms']), ratio(h['phase_launches'])
    a_gpu, a_launch = ratio(a['phase_gpu_ms']), ratio(a['phase_launches'])
    h_wall = h['wall_ms']['round'] / h['wall_ms']['plain']
    a_wall = a['round_cost_in_plain_steps']
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars(ax, ['H200 + быстрый CPU\n(шаг упирается в GPU)', 'A100, VM DataSphere\n(шаг упирается в запуск ядер)'],
         [[h_gpu, a_gpu], [h_launch, a_launch], [h_wall, a_wall]], [OURS_OLD, '#C9CED4', OURS],
         ['по работе GPU', 'по числу запусков ядер', 'замер (реальное время)'])
    ax.axhline(1.40, color=PAPER, ls='--', lw=1)
    ax.text(-0.45, 1.405, 'статья ≈ 1,40', va='bottom', ha='left', fontsize=8, color='#555555')
    ax.set_ylim(1.0, 1.85)
    ax.set_ylabel('цикл EAGLE, обычных шагов')
    ax.legend(loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig11_cycle_regimes.png')
    plt.close(fig)


if __name__ == '__main__':
    fig_3a()
    fig_depth()
    fig_cycle()
    fig_training()
    fig_setups()
    fig_trees()
    fig_budget()
    fig_regimes()
    print('figures written to', FIGURES)
