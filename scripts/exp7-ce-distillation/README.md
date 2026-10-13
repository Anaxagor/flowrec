# TIGER: CE distillation из распределения наград

Аналогичный эксперимент для GPT-2 с prefix layout из causal-окон:
[README_GPT2.md](README_GPT2.md).

TIGER на Yambda 50M/500M, полный last-128 layout, smooth curriculum и 3 выборки:
[README_YAMBDA.md](README_YAMBDA.md).

Эксперимент из §5.4 `_WWW_2026__flowrec.pdf`: дообучение TIGER из exp2
на Beauty 2014 и Instruments 2018 с потерей

\[
L = L_{SFT} + \lambda_D\,\mathbb E_x[-\sum_{i\in I}q_x(i)\log\pi_\theta(i\mid h)].
\]

В соответствии с выбранным режимом по умолчанию CE оценивается по **10 независимым
выборкам с возвращением непосредственно из полного распределения `q`**.
Это несмещённая оценка указанной суммы. Число выборок меняется через
`--num-samples`; `--estimator exact` вычисляет всю сумму по каталогу.

Награда и её численные ограничения совпадают с текущим matched exp4:

```text
r = max(1e-6, sasrec * interaction * hierarchical_similarity)
R = max(1e-6, r ** power)
q = R / sum_over_entire_catalog(R)
power in {0.25, 1, 4, 16}
```

Компоненты имеют вес 1 и те же собственные epsilon floors, что в exp4.
`sasrec` — sigmoid от оценки замороженного exp2 SASRec (temperature 1),
`interaction` — свидетельства строго до времени обучающего target,
`hierarchical_similarity` — близость к обучающему target по RQ-VAE codebooks.
**Степень применяется ко всему произведению.** При power=16 финальный floor
может существенно менять распределение; он намеренно сохранён для сравнения.

Нормировка `q` включает весь каталог, включая ранее просмотренные items.
В исходном режиме `--sampling reward` нет ограничения поддержки выборки,
подмешивания target или отбора top-K. `pi` использует локальную нормировку по допустимым продолжениям SID,
как у текущего GFlow и в уравнении (3) статьи. CE суммирует token log-probabilities
внутри SID. SFT остаётся обычным token CE по полной лексике. Обе части обучают
encoder; его выход переиспользуется для кандидатов.

## Протокол

Скрипт читает `configs/exp2-tuned-params/finetune_base_matched.json` и использует
проверенные функции из `gflow_finetune_tiger_matched.py`:

- исходные веса `checkpoints/exp2-tuned_params/initial/tiger/<dataset>/best.pt`;
- все обучающие префиксы exp2: Beauty — 131 413, Instruments — 116 989;
- каталоги 12 101 / 8 556 items, те же разбиения и semantic IDs;
- batch 32 × accumulation 64 = 2048, AdamW, lr 3e-5, weight decay 0.01;
- validation после каждой эпохи, монитор NDCG@5 с beam 5; начальный checkpoint
  участвует в выборе best, patience 5 после первых 10 эпох, максимум 500 эпох;
- финальная оценка выбранного best: полные validation/test, beam 20,
  HR/NDCG/Coverage/ARP при K=5,10,20;
- `rho in {1, 0.1}`, как в exp4: один раз на исходном checkpoint вычисляется
  `lambda_D = rho / median(||grad CE|| / ||grad SFT||)` по пяти батчам в FP32.
  Коэффициент сохраняется и не пересчитывается при resume.

Основная сетка: **2 datasets × 4 powers × 2 rho × seed 42 = 16 запусков**.
`--seeds 42 43 44` расширяет её до 48. Порядок: dataset, power, rho, seed.
В исходном режиме `--sampling reward` первые 10 эпох сохраняют общий stopping
protocol, но CE не использует curriculum.
Reward lookup никогда не вызывается на validation/test: target-зависимая
награда используется только для обучения.

## Exact и абляции с двумя фазами curriculum

Дополнительная сетка состоит из двух частей:

- **16 exact-запусков** для произведения всех трёх наград: те же два датасета,
  четыре степени, два rho и seed 42, точная сумма CE по всему каталогу.
- **96 sampled-запусков**: по отдельности `sasrec`, `interaction`, `similarity`
  и пары `sasrec_x_interaction`, `sasrec_x_similarity`,
  `interaction_x_similarity`, с теми же степенями/rho/seed и curriculum.

Для каждой комбинации `R = max(1e-6, max(1e-6, product(selected_rewards)) ** power)`.
Curriculum использует **это распределение наград**, а не отдельного SASRec-семплера.
Для каждого обучающего примера сортируем весь каталог по убыванию `q`:
первая половина содержит `floor(catalog_size / 2)` items с высокой наградой,
вторая — оставшиеся items с низкой. При равенстве наград сохраняется порядок
каталога. Половины определяются числом items, а не суммарной вероятностью.

1. Первые `--curriculum-epochs` эпох (по умолчанию 10): только нижняя половина.
2. Начиная со следующей эпохи: только верхняя половина.

В обеих фазах делаем `--num-samples` независимых выборок **с возвращением**,
с вероятностями исходного `q`, перенормированными внутри текущей половины:

```text
p_phase(i) = q(i) / sum(q(j) for j in current_half),  i in current_half
p_phase(i) = 0,                                     otherwise
CE_phase = mean(-log pi(sampled_item | history))
```

В режиме `--sampling curriculum` плавного перехода и весов `q/p` нет. Поэтому это CE условного распределения
текущей половины, а не несмещённая оценка CE полного `q` в каждой фазе.
При `--curriculum-epochs 0` сразу используется верхняя половина.
Число эпох переводится в число обучающих microbatches с учётом ограничения
длины эпохи; resume сохраняет момент переключения. Калибровка lambda использует
распределение первой фазы. SFT и протокол оценки остаются прежними.

## Плавный curriculum

Отдельный режим `--sampling curriculum_smooth` постепенно переносит вероятность
выбора с нижней половины на верхнюю. Разбиение каталога и пропорции `q` внутри
каждой половины такие же, как у двухфазного режима:

```text
T = ceil(curriculum_epochs * training_microbatches_per_epoch)
alpha = clip(global_microstep / T, 0, 1)  # при T=0: alpha=1
p(i) = (1-alpha) * q_low_normalized(i) + alpha * q_high_normalized(i)
```

В начале вероятность выбрать нижнюю половину равна 1, в середине перехода
каждая половина выбирается с вероятностью 0.5, после перехода выбирается только
верхняя. Вес меняется после каждого обучающего microbatch. Сначала нормируется
`q` отдельно внутри каждой половины, затем смешиваются эти два распределения:
исходная суммарная масса верхней половины не подавляет первую фазу.

В этом режиме `--curriculum-epochs` означает **длительность линейного перехода**
от начала обучения, без предварительной выдержки на нижней половине. Например,
при значении 5 к началу шестой эпохи выборка идёт только из верхней половины.
По умолчанию длительность 10 эпох; 0 сразу включает верхнюю половину.
Выборка с возвращением и обычный CE без `q/p`: целевое распределение меняется
вместе с `alpha`. Калибровка lambda выполняется при `alpha=0` (при T=0 — 1).
Resume восстанавливает global microstep, поэтому переход продолжается с того же
места. Лог `distill/curriculum_high_probability` показывает текущий вес верхней
половины (в эпохальных средних — его среднее за обработанные батчи).

## Запуск на кластере

Из корня репозитория:

```bash
bash scripts/slurm/submit_ce_distill_tiger.sh --dry-run
bash scripts/slurm/submit_ce_distill_tiger.sh
```

Для двух дополнительных массивов (exact + шесть curriculum-абляций):

```bash
bash scripts/slurm/submit_ce_distill_tiger.sh --extensions --dry-run
bash scripts/slurm/submit_ce_distill_tiger.sh --extensions
# Например, 5 эпох на нижней половине, затем верхняя:
bash scripts/slurm/submit_ce_distill_tiger.sh --extensions -- --curriculum-epochs 5
```

`--extensions` готовит кеши один раз, затем ставит два массива: 16 exact и
96 curriculum задач. `--seeds 42 43 44` расширяет их до 48 и 288 задач.
Оба массива ждут успешного завершения подготовки кешей. Если совместимые кеши
уже готовы, добавьте `--skip-cache` перед `--`.

Для тех же шести абляций с плавным переходом используйте обычный helper:

```bash
# 96 задач на Beauty/Instruments; добавьте --dry-run перед -- для просмотра.
bash scripts/slurm/submit_ce_distill_tiger.sh -- \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

Без `--combos` запускается стандартная сетка из 16 ячеек с тремя наградами.
`--extensions` по-прежнему выбирает exact и резкий двухфазный curriculum;
плавный режим подаётся отдельно приведённой командой.

Для подачи предварительная активация окружения не обязательна: helper ищет
`python` / `python3` версии ≥3.9, при необходимости загружает `module load python`
и пробует активировать `flowrec`. Подсчёт сетки использует только стандартную
библиотеку. В вычислительных задачах `flowrec` активируется отдельно.
`CONDA_ENV` меняет имя окружения; явный `PYTHON_BIN` применяется и при подаче,
и внутри задач, поэтому должен указывать на Python с зависимостями проекта.
Для старой копии helper или ручного просмотра сетки:

```bash
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowrec
python scripts/exp7-ce-distillation/ce_distill_tiger.py --list
```

Helper без `--dry-run` ставит в Slurm три массива с зависимостями:

1. SASRec factors: 2 задачи, по одному V100 на датасет.
2. Interaction и hierarchical factors: 4 задачи на CPU.
3. Обучение: 16 задач, каждая ждёт успешного завершения обоих cache arrays.

Выбор ресурсов: небольшому TIGER и 32×10 кандидатам достаточно одного
**V100 32 GB**; GPU нужен для обучения и SASRec encoding. Для CPU-компонентов
награды GPU не выделяется. `exact` собирает массу `q` на общих префиксах SID и
вычисляет CE каждого ветвящегося префикса один раз; детерминированные продолжения
имеют log-probability 0. Это та же полная сумма, без семплирования или отсечения
каталога. Decoder работает чанками с gradient checkpointing. Exact всё равно
требует существенно больше вычислений, чем 10 выборок на пример. Это оценка потребности,
а не измерение производительности на кластере; проверьте загрузку после первого
запуска и при необходимости измените batch/accumulation вместе.

Учтены [правила кластера](https://hpc.hse.ru/llms.txt): partition `rocky`,
account `proj_1752`, GPU constraint `type_a|type_c`, CPU constraint `type_d`,
без номера узла и `--mem`. Типы и характеристики проверены по
[официальному описанию оборудования](https://hpc.hse.ru/hardware/hpc-cluster).
Обучение получает 23:59, GPU-кеш 2 часа, CPU-кеш 4 часа;
`TRAIN_TIME`, `EXACT_TIME`, `CACHE_GPU_TIME`, `CACHE_CPU_TIME` позволяют изменить лимиты.
`MAX_PARALLEL=8` ограничивает одновременные задачи каждого массива.
Для exact в `--extensions` размер чанка 64 (`EXACT_CATALOG_CHUNK_SIZE`),
checkpoint сохраняется после каждого optimizer update.

```bash
# Повторы; доступны также ACCOUNT=proj_1925 / proj_1825.
bash scripts/slurm/submit_ce_distill_tiger.sh -- --seeds 42 43 44

# Малый pilot с отдельными output/cache, коротким лимитом и двумя ячейками.
TRAIN_TIME=00:30:00 bash scripts/slurm/submit_ce_distill_tiger.sh -- \
  --smoke --powers 1 --rho-targets 1

# Другая оценка ТОГО ЖЕ CE; отдельные директории от sampled.
bash scripts/slurm/submit_ce_distill_tiger.sh --skip-cache -- \
  --datasets beauty --powers 1 --rho-targets 1 --estimator exact

# Отдельная curriculum-абляция, первые 5 эпох — items с низкой наградой.
bash scripts/slurm/submit_ce_distill_tiger.sh --skip-cache -- \
  --sampling curriculum --curriculum-epochs 5 --combos interaction_x_similarity
```

`--skip-cache` допустим, когда совместимые кеши нужных компонентов уже существуют.
Сам training runner не строит отсутствующие кеши: так GPU-задача не ждёт
длительной CPU-подготовки. Кеши по умолчанию переиспользуются из matched exp4;
для независимых кешей передайте `--cache-root <parent>` всем фазам.
Проверяются manifest, количество строк, соответствие компонентов и порядок SID.
Скрытые состояния SASRec в существующих кешах хранятся в float16: exact означает
точную сумму восстановленных кешированных наград, а не отсутствие квантования кеша.

## Локальная проверка и результаты

Для короткой проверки на машине с зависимостями, датасетами и exp2 assets:

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger.py \
  --datasets beauty --cache-only --smoke --device cpu --num-workers 0
python scripts/exp7-ce-distillation/ce_distill_tiger.py \
  --datasets beauty --powers 1 --rho-targets 1 --index 0 \
  --smoke --device cpu --num-workers 0
```

Smoke ограничивает обучение 256 примерами / 8 батчами, evaluation двумя батчами
и отделяет все outputs/caches суффиксом `_smoke`. На login-узле такие вычисления
не запускают: используйте Slurm pilot выше. `--list`, `--count` и
`--aggregate-only` читают только метаданные и не импортируют ML-библиотеки.

Основной каталог:
`experiments/exp7-ce-distillation/<dataset>/sampled_n10/pow_<p>/rho_<rho>/seed_<seed>/`.
Точная сумма трёх наград попадает в отдельную ветку `exact/`. Абляции сохраняются в
`experiments/exp7-ce-distillation/<dataset>/<combo>/curriculum_n10_epochs10/pow_<p>/rho_<rho>/seed_<seed>/`.
У плавного режима отдельная ветка `curriculum_smooth_n10_epochs5/` (для E=5).
Число выборок и длительность первой фазы/перехода входят в имя ветки. В `artifacts/`
остаются переиспользуемые reward factors; результаты обучения лежат в `experiments/`.
Каждая ячейка сохраняет:
`identity.json`, `resolved_config.json`, `calibration.json`, `training_log.jsonl`,
`training_history.json`, `checkpoints/{best,last}.pt`, `result.json`.
Лог включает CE/SFT, энтропию teacher, максимальную вероятность и вероятность
наблюдённого target. Отдельно сохраняются init baseline, время подготовки кешей,
число шагов и длительность обучения/оценки. Подготовка teacher не исчезает из
учёта затрат при использовании готового кеша: reuse указан в cache timing.

После завершения массива:

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger.py --aggregate-only
# Если меняли сетку/режим, укажите те же аргументы, например --seeds 42 43 44.

# Дополнительные массивы собираются отдельно:
python scripts/exp7-ce-distillation/ce_distill_tiger.py --aggregate-only --estimator exact
python scripts/exp7-ce-distillation/ce_distill_tiger.py --aggregate-only \
  --sampling curriculum --combos sasrec interaction similarity \
  sasrec_x_interaction sasrec_x_similarity interaction_x_similarity
python scripts/exp7-ce-distillation/ce_distill_tiger.py --aggregate-only \
  --sampling curriculum_smooth --curriculum-epochs 5 \
  --combos sasrec interaction similarity sasrec_x_interaction \
  sasrec_x_similarity interaction_x_similarity
```

Создаются `sampled_n10/metrics.csv` и `sampled_n10/summary.json`; summary показывает
пропущенные ячейки и выбирает конфигурацию по среднему **validation NDCG@5**
по всем запрошенным seeds. Test не используется для выбора конфигурации.
Для дополнительных массивов сводки находятся в `exact/` и
`ablations/curriculum_n10_epochs10/`. В абляциях лучшая степень/rho выбираются
отдельно для каждой комбинации наград и датасета.
Сводка плавного режима с E=5 сохраняется в `ablations/curriculum_smooth_n10_epochs5/`.

Повторная подача продолжает `last.pt`, сохраняя optimizer, AMP, раннюю остановку
и шаг curriculum; завершённые ячейки пропускаются. Для dropout и shuffle
применяется `seed + epoch`. `--checkpoint-every-updates N` дополнительно сохраняет
атомарный checkpoint после каждых N optimizer updates; `0` оставляет сохранение
в конце эпохи. При восстановлении внутри эпохи пропускаются уже обработанные
батчи и восстанавливается RNG. Для exact по умолчанию N=1, для sampled N=0.
Средние train-метрики возобновлённой эпохи относятся только к её оставшимся
батчам; validation всегда выполняется с обычным заданным объёмом данных.
Если обучение уже завершено, но job прерван при итоговой оценке, повторяется
только оценка. Изменение протокола/assets в существующем run dir вызывает ошибку;
для нового протокола задайте новый `--output-root`.

Сравнивайте с исходным TIGER и matched continued-SFT, а также с трёхкомпонентными
ячейками GFlow при тех же power/rho. У старых GFlow-запусков sampler имеет другую
поддержку и curriculum, поэтому одно такое сравнение не изолирует только геометрию
loss. Для строгого вывода §5.4 также нужны одинаковая поддержка выборки и отдельные
сравнения при равном числе updates и равных вычислительных бюджетах.
