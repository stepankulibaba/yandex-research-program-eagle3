# Исправления по ревью

| Находка | Реализация |
|---|---|
| Удаление готового обучения | Нет recursive reset; отдельные main/smoke; manifest + хеш + validator |
| Скрытые ошибки Bash/pipeline/check | Тонкие strict-shell launchers; subprocess return codes/timeout; итог FAILED |
| Частичные загрузки HF | Pinned revision, полный manifest, размеры, индексы, safetensors header |
| Чужой сервер/оставшиеся workers | Проверка свободного порта, server_info, own process groups, cleanup SIGTERM/timeout |
| CUDA toolkit/JIT | Explicit Triton/PyTorch backends, kernel gate T=1, C/C++ toolchain checks, scoped DS stub |
| Частичные benchmark-файлы | Полный набор ID/ходов, валидные числа, atomic SGLang JSON, commit marker |
| Разные training prompts/padding | Общий авторский CPU preprocessing, canonical IDs/masks, fixed2048 |
| Dtype/checkpointing | Согласованный BF16 throughput protocol, checkpointing off |
| Разный scheduler | Constant LR в обоих speed adapters; paper training не выдаётся за неизменённый |
| Разные warm-up/окна/шаги | Optimizer-step warm-up, CUDA boundaries, точные counters/ms-step |
| SGLang τ | Раздельные verification tokens/round и completion/round; zero-round обработка |
| 3a/3b stopping/timing | Авторский 3a сохранён; bounded HTTP протокол описан отдельно, explicit EOS/eot/cleanup |
| T=1 | Явные параметры/seed, прогрев при нужной T, три seed, запрет greedy fallback |
| Непривязанные версии/FA2 подписи | Pinned repositories/models/SGLang, actual env fingerprint, реальный backend в таблице |
| Нерепрезентативный бюджет | Два источника/группы длин/multi-turn/shared policy; reserve/sensitivity; sunk-cost формула |
| Поздние важные стадии/7h | Author T=0/T=1 и базовое обучение раньше; общий дедлайн; частичные отчёты |
| Smoke speedup gate/память | Нет требования speedup>1; allocated/reserved PyTorch peaks, scope явно подписан |
| Дополнительно: накопление градиента | В adapted original удалён model.zero_grad на каждом micro-batch |

## Локальная проверка

- 15 CPU-only regression tests: PASS.
- AST всех Python исходников: PASS.
- bash -n пяти launchers: PASS.
- UTF-8 без BOM, LF: PASS.
- Точки адаптации/сигнатуры pinned NeMo recipe проверены по upstream source.
- GPU, kernels, FP8 backward, реальный FA2 и скорость не запускались.
- Численный/градиентный паритет двух реализаций TTT не подтверждён.
- Полный census ShareGPT/UltraChat не выполнен: бюджет остаётся предварительным
  сценарием чувствительности, а не доказательством достаточности 100 GPU-hours.

## Следующий запуск на сервере

1. python3 orchestrate.py setup — установка/скачивание до GPU allocation.
2. bash check.sh — GPU smoke, отдельная директория результатов.
3. NIGHT_SECONDS=25200 bash night.sh — основной прогон в восьмичасовой allocation.

При FAILED проверить logs/main либо logs/smoke и stages.json.
Повторный запуск сохраняет и использует подходящие завершённые стадии.

## Проверка после ревью (9 октября)

Сверено с исходниками на закреплённых версиях: все пины (EAGLE, NeMo, модели, датасеты, SGLang 0.5.9) существуют;
методы NeMo, которые подменяет `train_nemo.py`, и якоря в `main.py` авторов на месте; `/server_info`, `sampling_seed`,
`TREE_SPEC_KERNEL_AVAILABLE` и Triton-бэкенд для top-k > 1 есть в SGLang 0.5.9.

Исправлено:
- **SGLang, счётчик принятых токенов.** В 0.5.9 поле называется `spec_accept_token_num`, а не `spec_accepted_tokens`;
  при нуле раундов SGLang не отдаёт счётчики вовсе. Раньше каждый EAGLE-3 замер в SGLang падал.
- **Одна ошибка обрывала ночь.** Теперь упавшая стадия получает FAILED, остальные идут дальше; останавливают
  только общий дедлайн и сигнал. В конце — список упавших стадий.
- **Таймаут бенчмарка** 30 → 90 мин (`BENCH_TIMEOUT`): авторская обычная генерация на MT-bench могла не успеть.
- **Скачивание:** уже проверенная модель проверяется локально, без обращения к Hub; таргет берётся из
  `../nemo_speed/models` (симлинк) вместо повторных 16 ГБ.

## Глубина дерева (9 октября)

Скрипты оценки авторов по умолчанию берут `--depth 5` — дерево EAGLE-2 (глубина 6 в статье). Для EAGLE-3 статья
указывает глубину 8 при тех же 60 узлах, это `--depth 7` в коде (значение по умолчанию в `EaModel.from_pretrained`).
Ночной 3a посчитан на `--depth 5`. Теперь глубина передаётся явно: `--total-token 60 --depth 7 --top-k 10`;
дерево SGLang «как в статье» — 8/10/60; 6/10/60 оставлено как дерево EAGLE-2 для сравнения.

## Читаемая версия (9 октября)

Код переписан для людей без изменения поведения: `runtime.py` → `common.py`, `training_data.py` → `training_common.py`
(вместе с секундомером `SpeedWindow`), `orchestrate.py` разбит на понятные стадии. Проверка: старая и новая
версии на фейковом сервере выдают одни и те же 178 команд в том же порядке (режимы night, check, train), тесты проходят.
Изменено намеренно: (1) зависание сервера SGLang при старте — теперь ошибка стадии, а не конец ночи;
(2) хеши файлов кода больше не входят в настройки результата (правка комментариев не пересчитывает готовое);
вместо этого `common.SCHEMA` (сейчас 4) повышается при изменении того, что измеряется.
