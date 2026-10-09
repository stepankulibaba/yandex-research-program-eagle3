# Задача 4: скорость шага обучения (полная таблица)

Прогон 9 октября 2026, одна H200 NVL. Вывод `nemo_speed/artifacts/main/summary.txt` без изменений.

- **Данные:** 1000 примеров `frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct`, токенизированных функцией авторов, и общий словарь драфта на 32K.
- **Замер:** по шагам оптимизатора после 20 шагов прогрева, TTT = 7.
- **Режимы:**
  - `matched-bf16-fixed2048-v1` — одинаковая работа для обоих тренеров: BF16, строки по 2048 токенов, постоянный lr, без gradient checkpointing;
  - `author-as-is-fp16-v1` — тренер авторов в своих настройках: fp16, checkpointing, без паддинга, их расписание lr.
- **Чем стеки отличаются и после выравнивания:** детали loss, RoPE драфта, точность оптимизатора, версии библиотек (torch 2.5.1 у авторов, 2.10 у NeMo). Поэтому сравнивается скорость стеков, а не одинаковое обучение.
- **Packing** меняет содержимое строки, поэтому сравнивать надо документы/с и токены/с, а не время шага.

| run | protocol | draft attention | batch × accumulation | packing | ms / optimizer step | documents/s | non-pad tok/s | padded tok/s | supervised tok/s | peak allocated / reserved GiB | measured steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo_compile | matched-bf16-fixed2048-v1 | eager | 1 × 2 | 2048 | 403.64 | 15.494 | 8454.2 | 10108.8 | 4582.2 | 28.97 / 30.45 | 130 |
| nemo_compile_fp8 | matched-bf16-fixed2048-v1 | eager | 1 × 2 | 2048 | 397.88 | 15.718 | 8576.5 | 10254.9 | 4648.5 | 28.34 / 29.85 | 130 |
| nemo_eager_nopack | matched-bf16-fixed2048-v1 | eager | 1 × 2 | 0 | 461.07 | 4.338 | 2258.7 | 8883.7 | 1181.2 | 31.59 / 33.06 | 480 |
| nemo_eager_pack | matched-bf16-fixed2048-v1 | eager | 1 × 2 | 2048 | 465.81 | 13.426 | 7325.8 | 8759.5 | 3970.6 | 31.59 / 33.06 | 130 |
| original | matched-bf16-fixed2048-v1 | eager | 1 × 2 | 0 | 635.61 | 3.147 | 1638.4 | 6444.2 | 856.9 | 36.55 / 40.74 | 480 |
| original_asis | author-as-is-fp16-v1 | eager | 1 × 2 | 0 | 148.46 | 13.472 | 6924.5 | 6924.5 | 3545.9 | 28.38 / 30.42 | 480 |

`nemo_fa2_pack` и `nemo_fa2_pack_mb4` не запускались: flash-attn 2 не ставится без CUDA toolkit.
