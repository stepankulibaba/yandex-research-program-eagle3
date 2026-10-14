"""Build research figures exclusively from the checked-in measurements."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results/qwen3-1.7b'
FIGURES = ROOT / 'figures'

def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    analysis = json.loads((RESULTS / 'factorial_analysis.json').read_text())
    names = ['train24hi', 'train24lo', 'train96hi', 'train96lo']
    labels = ['24 примера\nскорость 1e−4', '24 примера\nскорость 1e−5', '96 примеров\nскорость 1e−4', '96 примеров\nскорость 1e−5']
    effects = [analysis['primary']['conditions'][name]['relative_initial'] for name in names]
    values = [(r['geometric_ratio'] - 1) * 100 for r in effects]
    errors = [[(r['geometric_ratio'] - r['bootstrap_interval'][0]) * 100 for r in effects],
              [(r['bootstrap_interval'][1] - r['geometric_ratio']) * 100 for r in effects]]
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    ax.bar(labels, values, color=['#b54c44', '#317567', '#b54c44', '#317567'], yerr=errors, capsize=5, width=.58)
    ax.axhline(0, color='#555555', linewidth=.8)
    for i, value in enumerate(values):
        ax.text(i, value + (5 if value > 0 else -5), f'{value:+.1f}%'.replace('.', ','), ha='center', va='center', fontweight='bold')
    ax.set_ylim(-40, 24)
    ax.set_ylabel('Изменение принятия, %')
    ax.set_title('Дообучение: одинаковый бюджет 79 712 входных токенов')
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .01, 'Qwen3-1.7B + AngelSlim EAGLE-3 · 32 новых запроса UltraChat · описательные 95%-интервалы', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1));fig.savefig(FIGURES / 'factorial_acceptance.png', dpi=170);plt.close(fig)
    rows = json.loads((RESULTS / 'memory_measurements.json').read_text())
    lengths = [166, 256, 384, 512, 640, 896, 1024]
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    for mode, label, color in [('dense', 'Копии сдвигов', '#b54c44'), ('split', 'Общий буфер', '#317567')]:
        peaks = [statistics.median(r['full_peak_mib'] for r in rows if r['kind'] == 'memory' and r['mode'] == mode and r['length'] == n) / 1024 for n in lengths]
        ax.plot(lengths, peaks, 'o-', label=label, color=color)
    ax.set_xlabel('Длина последовательности, токены');ax.set_ylabel('Полный пик выделенной памяти, ГиБ')
    ax.set_title('Общий буфер вероятностей цели уменьшает пик памяти')
    ax.legend();ax.grid(alpha=.18);ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .01, 'T4 · пакет 2 · TTT 7 · FP16 / основные веса FP32 · синтетические документы', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1));fig.savefig(FIGURES / 'memory_peaks.png', dpi=170);plt.close(fig)
    print('Figures rebuilt from saved measurements: 2')

if __name__ == '__main__':
    main()
