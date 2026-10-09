# Скорость обучения EAGLE-3

`run.sh` вызывает единый оркестратор из соседней `../inference_speed`.
Готовые результаты сохраняются в `artifacts/main`; smoke — в `artifacts/smoke`.
Повторный запуск проверяет manifest/хеши/полноту и пропускает завершённые стадии.

```bash
SMOKE=1 bash run.sh
bash run.sh
```

Полный протокол, ограничения и подготовка окружений:
[../inference_speed/README.md](../inference_speed/README.md).

Оригинальный тренер адаптирован к согласованному BF16 throughput-тесту:
общие токены и маски, vocab mapping, fixed2048, constant LR, без activation
checkpointing. Это не измерение неизменённого FP16 training recipe статьи.
Результаты содержат фактический ms/optimizer-step и счётчики измеренного окна.
