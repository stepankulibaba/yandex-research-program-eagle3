# Инструкция для агентов

Этот файл — для ИИ-агентов и всех, кто дописывает репозиторий. README пишутся для людей; технические правила собраны здесь.

## Структура

```
<направление>/
  README.md        для людей: вопрос → что сделали → что получилось → что это значит → что дальше
  code/            код; скрипты считают корнем папку направления, а не code/
  results/<модель>/  json и логи (небольшие)
  figures/         png, строятся скриптами code/make_*.py
  reports/         PDF-отчёты
```

- Новое направление — новая папка по этому шаблону и строка в таблице корневого `README.md`.
- Чужие направления (NeMo и т. п.) не трогать без просьбы владельца.

## Пути

- В `code/` пути к `models/`, `data/`, `EAGLE/`, `results/` считаются от **родительской** папки: `ROOT = Path(__file__).resolve().parent.parent`, а `sys.path` указывает на `code/`. После переноса или добавления скриптов проверить, что `make_*.py` отрабатывают и пишут в `figures/`.
- `layer-selection/router/code/run_dsl.sh` сначала делает `cd` в папку направления и вызывает `code/*.py`.
- В `layer-selection/frozen/code` и `layer-selection/router/code` есть `original/` — код, на котором реально шли прогоны. Рядом — читаемая версия, проверенная на совпадение результатов (побайтно). Править читаемую версию; `original/` не трогать (кроме путей).
- `big-training/code` на сервере лежит как `~/data/big_training` рядом с `inference_speed` и `nemo_speed` из `reproduction/code`: `settings.py` считает корнем родителя этой папки. Скрипты `make_*.py` в ней — как везде: корень = `big-training/`.
- `extra-checks/code/zoo_eval.py check|validate` используют модули из `layer-selection/frozen/code` и его папку моделей.
- `reproduction/code/inference_speed` и `reproduction/code/nemo_speed` должны лежать рядом: оркестратор (`orchestrate.py`) считает свою папку корнем, а `../nemo_speed` — папкой обучения. Окружения, клоны (`EAGLE/`, `Automodel/`), модели и результаты создаются внутри этих папок на GPU-сервере и в git не попадают.

## Что нельзя коммитить (репозиторий публичный)

- Модели, данные, чекпоинты, `.venv`, архивы — уже закрыты в `.gitignore`. Исключения: `results/**/final_eval.pt` и `results/**/*.log`.
- Адреса серверов, логины, имена пользователей в путях (`/home/<user>/…` → `~/…`), токены, ключи, почты, ники на Kaggle и HF.
- Проверка перед коммитом: `git grep -nE "/home/|@[a-z0-9.-]+\.(ru|com)|hf_|gho_|ssh-"`.

## Метрики (не путать)

- **τ по протоколу статьи** — реальная генерация EAGLE деревом: total_token 60, **depth 7**, top_k 10, temperature 0; `max_new_tokens` фактически 512. Глубину передавать явно: скрипты оценки авторов по умолчанию берут `--depth 5` — это дерево EAGLE-2 (глубина 6 в статье); для EAGLE-3 статья даёт глубину 8 = `--depth 7` (так же по умолчанию в `EaModel.from_pretrained`). В SGLang `--speculative-num-steps` — это длина ветки, т. е. `depth + 1` = 8. Прежние сравнения со статьёй (`extra-checks`, DSL-8B) сделаны на `--depth 5`.
- **Цепочечная τ** — офлайн, одна цепочка глубины 6 по greedy-тексту target (`router/code/metrics.py`). Совпадает с EAGLE в режиме цепочки с точностью 0,01, но ниже τ по дереву. Абсолютные значения двух метрик не сравнивать — только приросты.
- Интервалы — парный bootstrap по вопросам.

## Известные ловушки

- Индекс слоя i — это **вход** слоя i (`hidden_states[i]` в HF); N — выход последнего слоя **до** финальной нормы (берётся хуком). У драфтов DeepSeek `target_layer_ids` в другой нумерации (+1).
- У драфта DSL нет `rope_theta` в конфиге → брать 10000 (как в EAGLE), а не 1e6.
- Для Llama-3 стоп-токены — `eos` и `<|eot_id|>`.
- Пословный роутер: при входе размерности H и lr 5e-3 выбор застывает в первые десятки шагов. Использовать `token_top*_s` (вход / √H, lr 1e-3). Результаты `token_top3/2` на Qwen3-1.7B получены со старой версией — помечать как предварительные.
- Обучаемый драфт стартует с весов, обученных под оригинальные слои: у оригинальной тройки есть фора.
- `router/code/` (читаемая версия) побайтно эквивалентна `router/code/original/` — после правок в `draft.py`, `fusion.py`, `train.py` прогнать `check_equivalence.py` или аналогичный тест на маленьких моделях.
- Во frozen-коде этап обучения входов (`train_fusions.py`) грузит модель в float16, остальные — в bfloat16: так было в исходных прогонах.
- Порядок создания вариантов в `draft_inputs.build_variants` и `train_fusions.all_variants` задаёт случайную инициализацию — не переставлять.
- `reproduction`, сервер без CUDA toolkit (нет `nvcc`/`CUDA_HOME`): DeepSpeed 0.16.4 спрашивает `nvcc -V` и при установке, и при **каждом импорте** — ему даётся заглушка `venv_orig/stub_cuda` (печатает версию CUDA из torch); SGLang — с `SGLANG_ENABLE_JIT_DEEPGEMM=0`, Triton-attention и PyTorch-sampling (flashinfer может требовать JIT); flash-attn 2 не ставится.
- `reproduction`: в окружении SGLang 0.5.9 удалять пакет `kernels` (новая версия требует `huggingface_hub>=1.10`, transformers 4.57 держит <1.0 — сервер падает на старте). Счётчик принятых токенов в ответе SGLang — `meta_info.spec_accept_token_num`; при нуле раундов счётчиков нет вовсе.
- `reproduction`: скрипты с потоковым `datasets` падают при завершении Python (`PyGILState_Release`) уже после записи результатов — выходить через `os._exit(0)`.
- `reproduction`: SGLang грузит официальный драфт как обычную Llama, если в его `config.json` нет `architectures = ["LlamaForCausalLMEagle3"]`; оркестратор даёт SGLang копию конфига с этим полем (веса — ссылки) и `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
- `reproduction`: ShareGPT читать из файла `ShareGPT_V4.3_unfiltered_cleaned_split.json` напрямую (`hf_hub_download` + `json`): потоковый `datasets` портит записи. UltraChat `train_gen` — только промпты.
- `reproduction`: ускорение EAGLE зависит от машины, а не только от GPU: на виртуалке с медленным запуском ядер обычная генерация медленнее, и ускорение выше. Сравнивать машины — по ускорению **и** абсолютным ток/с.
- Зеркало `unsloth/Meta-Llama-3.1-8B-Instruct`: в `tokenizer_config.json` класс `PreTrainedTokenizer`; скрипты EAGLE (`use_fast=False`) с ним не получают токенизатор — менять на `PreTrainedTokenizerFast` (так у Meta).

- `big-training`: SGLang и NeMo нумеруют слои как «выход слоя i» — наши 16, 22, 29 у них 15, 21, 28 (`export.py`). `fc_norm` в SGLang 0.5.9 — только через внешний пакет `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_eagle31`; без него SGLang молча пропускает незнакомые веса. Серверу SGLang нужен `ninja` в `PATH` (он лежит в `venv_sglang/bin`).
- `big-training`: **SGLang 0.5.9 записывает в CUDA-графы слои цели по умолчанию** (2, N/2, N−3): `CudaGraphRunner` вызывает `set_eagle3_layers_to_capture()` без аргументов до того, как выставлены слои из конфига драфта, и те доходят только до prefill. Драфт с другими слоями тихо теряет τ (у ours −13 %). Пакет `sglang_eagle31` это исправляет; проверка — строка `target layers to capture (output ids) …, also in CUDA graphs` в логе сервера. Сравнивать SGLang с кодом авторов на каждом новом драфте.
- `big-training`: рецепт NeMo AutoModel для обучения драфта с нуля нестабилен (рост логитов q·k, см. `big-training/reports/eagle3_nemo_incident.pdf`); оба варианта учатся кодом авторов.
- `big-training`: τ-оценка на валидации — цепочка из 7 позиций, $1 + \sum_k \prod_{j\le k} acc_j$; ниже реальной τ на дереве, сравнивать только варианты между собой.

- `draft-staleness`: код авторов EAGLE с зеркалом `unsloth` получает вместо токенизатора `False` (медленный токенизатор
  Llama-3 не собирается) — после загрузки `EaModel` подставлять быстрый: `model.tokenizer = AutoTokenizer.from_pretrained(...)`.
- `draft-staleness`: при сравнении двух драфтов на одном target ответы НЕ всегда совпадают (bf16-проверка дерева иначе
  разрешает почти равные варианты); сравнивать τ на общем начале ответов (`analysis_fix_prefix.py`).
- `draft-staleness`: отображение «новые признаки → старые» по ridge подгонять как поправку к единичному
  (`Y ≈ X + [X,1]B`); ridge к нулю переобучается (4097 × 4096 параметров на слой при ~6 тыс. токенов).
- `draft-staleness`, Colab CLI: одно ssh-подключение на сессию; работа по ssh не считается активностью ядра, и сессия
  пропадает — нужен `code/colab/keepalive_pull.sh`; `colab exec`/`colab ssh` при пропавшей сессии молча создают
  новую CPU-сессию с тем же именем. Две задачи, которые качают и удаляют модели, одновременно не запускать.
- `draft-staleness`: `results/llama31-8b/geo1b_*_ids.npz` — компактная часть `geometry1b`; полные состояния (~34 МБ
  на модель) в git не хранятся.

## Проверки перед PR

1. `python -m py_compile` для всех изменённых `.py`, `bash -n` для `.sh`.
2. Скрипты графиков отрабатывают из новой раскладки.
3. Для изменений обучения — короткий прогон на маленьких случайных моделях (CPU/GPU ноутбука) и сравнение с прежней версией.
4. Обновить README направления и строку статуса в корневом README.

## Стиль README

- Для людей: простым языком, коротко, сначала вывод, потом детали. Один главный график на страницу.
- Формулы — через GitHub math (`$$ … $$`), а не сырым текстом в обратных кавычках.
- Числа — с запятой, проценты — к оригиналу.

## Текущее состояние

- `layer-selection`: Qwen3-1.7B и DSL-8B посчитаны, отчёты готовы.
- `extra-checks`: готово; сравнения со статьёй там на `--depth 5`.
- `big-training`: готово. baseline (2/16/29) и ours (16/22/29 + `fc_norm`) обучены до 546 600 диалогов (0,4 эпохи) и оценены в SGLang (с исправлением захвата слоёв) и кодом авторов. Все числа — `results/llama31-8b/results.json` и `tables.md`, графики — `code/make_results_figures.py`, данные собирает `code/collect_results.py` из копии логов сервера, дополнительные графики — `code/make_extra_figures.py`. Замер ours без исправления SGLang — `results/llama31-8b/final_eval/ours_sglang_capture_bug/`.
- `reproduction`: задачи 3a, 3b, 4, 5 посчитаны на H200, 3a (T = 0) и профиль повторены на A100 (DataSphere). Отчёты — `reproduction/reports/`, все числа отчётов — `reproduction/results/numbers.json`, графики строит `reproduction/code/make_report_figures.py`. LaTeX-исходники отчётов в репозиторий не входят. Драфт не обучали: стоимость обучения только оценена.
- `draft-staleness`: карта (10 target + 2×2 с R1-Distill), разложение, геометрия, поправка ранга 2 в `fc` проверена τ дерева на 4 моделях. Все числа — `draft-staleness/tables.md` и `results/llama31-8b/`, графики — `code/make_figures.py`, анализ — `code/analysis_*.py`. Отчёт и слайды с LaTeX-исходниками — `reports/`.
