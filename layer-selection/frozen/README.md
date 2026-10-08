# Замороженный драфт

Драфт не обучается (кроме входа fc в отдельных опытах). Смотрим, на какие слои он опирается и что даёт другой выбор слоёв.

| Файл | Что делает |
|---|---|
| `layer_lab.py` | общая библиотека: патч EAGLE для захвата любых слоёв, τ, пробы, CKA, варианты входа (fc, нормализация, смесь, MoE) |
| `run.py` | прогоны на H200: `eval_official` (абляции официальных драфтов, протокол статьи), `gen_data`, `analysis`, `fusion`, `sweep`, `lang` (языки, см. extra-checks) |
| `phase_a.py`, `phase_b.py` | те же опыты в варианте для Kaggle T4 (`build_notebooks.py` собирает ноутбуки) |
| `only_high.py`, `smoke.py` | отдельная абляция «только N−3» и быстрая проверка стенда |
| `make_figures.py`, `make_figures_h200.py` | графики в `figures/` |

```bash
python run.py eval_official dsl-8b      # аналогично qwen3-1.7b; модели в models/<pair>/{base,draft}, EAGLE/ рядом
python run.py gen_data dsl-8b && python run.py fusion dsl-8b && python run.py sweep dsl-8b
```

`results/h200/` — json и логи H200 (`rows_*.json` — τ по каждому вопросу для абляций), `results/kaggle_*` — Kaggle T4.
