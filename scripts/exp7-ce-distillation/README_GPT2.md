# GPT-2: CE distillation из наград на Beauty и Instruments

Runner: `scripts/exp7-ce-distillation/ce_distill_gpt2.py`.
Для парного сравнения обычной выборки из q и curriculum с поправкой q/proposal:
`bash scripts/slurm/submit_ce_distill_q_comparison.sh --profile gpt2 --skip-cache`.
Оба режима оценивают CE по q; [формулы, сетка и параметры](README.md#сравнение-двух-несмещённых-оценок-ce-по-q).
Начальные веса — exp2 GPT-2 `best_model.pt` для каждого датасета.
Цель, степени наград и калибровка аналогичны [эксперименту TIGER](README.md):
`L = L_SFT + lambda_D * CE`, `power in {0.25, 1, 4, 16}`,
`rho in {1, 0.1}`, seed 42. Награда — произведение выбранных компонентов
`sasrec`, `interaction`, `similarity` (hierarchical), степень применяется ко
всему произведению с epsilon floors из matched exp4. `q` нормируется по полному
каталогу GPT-2. `pi` нормируется по допустимым продолжениям semantic ID.

## Данные: prefix layout из causal layout

Используется тот же путь, что в
`gflow_finetune_gpt2_amazon_fullhist.py`: исходные `dense_causal` окна exp2
разворачиваются в `prefix_target`. Каждый активный target исходного окна
становится отдельным примером, сохраняя именно тот префикс, который был у него
в causal layout. На границе очередного окна история начинается заново;
предыдущие окна не дописываются к контексту.

Для Beauty и Instruments длина окна и stride равны 20 items. Каталог и полные
validation/test сохраняются из dense-разбиений. Используются все активные
обучающие targets без last-N сокращения. Аудит преобразования и его сигнатура
сохраняются вместе с конфигурацией; кеши проверяются на совпадение контекста,
подмножества, строк и порядка semantic IDs. Кеши TIGER для этого не подходят.

## Режимы CE

- `--estimator sampled --sampling reward`: 10 IID выборок с возвращением из
  полного `q`, несмещённая оценка полного CE.
- `--sampling curriculum`: первые `--curriculum-epochs` эпох выборка из нижней
  половины каталога по награде, затем из верхней. Внутри половины вероятности
  пропорциональны `q`.
- `--sampling curriculum_smooth`: линейная смесь нормированных половин
  `p_t = (1-alpha) * q_low + alpha * q_high`, где `alpha` растёт от 0 до 1 за
  `--curriculum-epochs` эпох, после чего используется верхняя половина.
- `--sampling curriculum_smooth_is`: смесь `m_t = (1-epsilon)*p_t + epsilon*q`,
  выборка из `m_t` и веса `q/m_t`; несмещённая оценка CE по q на всех эпохах.
  `--proposal-q-mix` задаёт epsilon, по умолчанию 0.1.
- `--estimator exact`: полная сумма CE по каталогу, без семплирования.

В старых режимах `curriculum` и `curriculum_smooth` CE относится к текущему
условному/смешанному распределению, без importance weights. Разбиение — по числу items, порядок
по убыванию награды, ties сохраняют порядок каталога. Длительность curriculum
по умолчанию 10 эпох; 0 сразу включает верхнюю половину. Шаг перехода —
обучающий microbatch; resume сохраняет прогресс.

Для CE история GPT-2 вычисляется один раз в дифференцируемый KV-кеш.
Кандидаты переиспользуют его с сохранением градиента через историю. SFT имеет
свой проход. Exact суммирует массу `q` на общих ветвящихся SID-префиксах;
детерминированные продолжения дают нулевой log-probability. Чанки используют
gradient checkpointing. Это точная сумма для кешированных наград, включая их
float16-квантование, а не отсечение каталога.

Matched-протокол: batch 32 × accumulation 64, AdamW lr 3e-5, weight decay 0.01,
validation каждую эпоху, начальный checkpoint участвует в выборе best.
Монитор — NDCG@5 с beam 5; patience 5 после первых 10 эпох, максимум 500 эпох.
Финальные validation/test: beam 20, HR/NDCG/Coverage/ARP при K=5,10,20.
`lambda_D` калибруется один раз по отношению норм градиентов CE/SFT в FP32.

## Slurm

Из корня репозитория на кластере. Плавный переход за 5 эпох, шесть одиночных
и парных наград, два датасета — **96 запусков**:

```bash
bash scripts/slurm/submit_ce_distill_gpt2.sh -- \
  --datasets beauty instruments \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

Exact для произведения трёх наград — **16 запусков**:

```bash
bash scripts/slurm/submit_ce_distill_gpt2.sh -- \
  --datasets beauty instruments --estimator exact \
  --combos sasrec_x_interaction_x_similarity \
  --catalog-chunk-size 64 --checkpoint-every-updates 1
```

Обе части с одной общей подготовкой кешей:

```bash
bash scripts/slurm/submit_ce_distill_gpt2.sh --extensions -- --curriculum-epochs 5
```

У GPT-2 helper `--extensions` выбирает именно **smooth + exact**.
Добавьте `--dry-run` перед `--`, чтобы вывести команды без подачи.
`--skip-cache` перед `--` допустим после подготовки совместимых GPT-2 factors.
`--seeds 42 43 44` расширяет обе сетки втрое. Каждая задача — отдельная
комбинация dataset/rewards/power/rho/seed; training arrays ждут cache arrays.

Оценка ресурсов: exp2 модели имеют 4 слоя, ширину 256 и контекст до 85 tokens;
обучение и SASRec encoding получают один V100 32 GB (`type_a|type_c`),
interaction/similarity кеши — CPU (`type_d`) без GPU. Exact ограничивает память
чанками и сохраняет `last.pt` после каждого optimizer update, чтобы длинную
эпоху можно было продолжить после лимита времени. Фактическая загрузка GPU
требует проверки при первом кластерном запуске.

Учтены [правила кластера](https://hpc.hse.ru/llms.txt) и
[описание оборудования](https://hpc.hse.ru/hardware/hpc-cluster): partition
`rocky`, account `proj_1752`, без `--mem` и привязки к номеру узла.
По умолчанию лимит обучения 23:59, SASRec cache 2 часа, CPU cache 4 часа;
`MAX_PARALLEL=8`. Доступны `ACCOUNT`, `TRAIN_TIME`, `EXACT_TIME`,
`EXACT_CATALOG_CHUNK_SIZE`, `CACHE_GPU_TIME`, `CACHE_CPU_TIME`.
Helper сам находит Python >=3.9 для метаданных; вычислительные задачи активируют
`flowrec`. Можно явно указать `PYTHON_BIN` с зависимостями проекта.

## Кеши, результаты и проверка

Новые GPT-2 factors:
`artifacts/exp7-ce-distillation-gpt2/<dataset>/reward_factors/<component>/`.
Внутреннее имя hierarchical similarity — `sim_hier`; публичный CLI использует
`similarity`, как у TIGER. `--cache-root` задаёт другой корень с тем же layout.

Результаты:

```text
experiments/exp7-ce-distillation-gpt2/<dataset>/exact/pow_<p>/rho_<r>/seed_<s>/
experiments/exp7-ce-distillation-gpt2/<dataset>/<combo>/curriculum_smooth_n10_epochs5/pow_<p>/rho_<r>/seed_<s>/
```

Сохраняются identity/config, аудит dense→prefix, calibration, логи, история,
`checkpoints/{best,last}.pt` и `result.json`. Разные модели и sampler-режимы
имеют отдельные директории. Повторная подача продолжает незавершённый запуск;
несовместимый протокол или assets вызывают ошибку вместо перезаписи результата.
Reward lookup используется только в обучении.

После завершения:

```bash
python scripts/exp7-ce-distillation/ce_distill_gpt2.py --aggregate-only --estimator exact
python scripts/exp7-ce-distillation/ce_distill_gpt2.py --aggregate-only \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

Сводки `metrics.csv` и `summary.json` появляются в `exact/` и
`ablations/curriculum_smooth_n10_epochs5/`. Степень/rho выбираются отдельно
для каждой комбинации наград по validation NDCG@5, без выбора по test.

Короткая локальная проверка (на машине с assets и зависимостями):

```bash
python scripts/exp7-ce-distillation/ce_distill_gpt2.py \
  --datasets beauty --cache-only --smoke --device cpu --num-workers 0
python scripts/exp7-ce-distillation/ce_distill_gpt2.py \
  --datasets beauty --powers 1 --rho-targets 1 --sampling curriculum_smooth \
  --curriculum-epochs 0.5 --smoke --device cpu --num-workers 0
```

Smoke использует отдельный output с суффиксом `_smoke`, кеши в его `cache/`,
8 обучающих и 2 evaluation батча. На login-узле вычисления не выполняются;
для кластерной проверки передайте `--smoke` helper и короткий `TRAIN_TIME`.
`--list`, `--count`, `--aggregate-only` используют только стандартную библиотеку.
