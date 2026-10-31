# TIGER: CE distillation на Yambda 50M и 500M

> **Доступность:** scheduler-запускаторы ниже — локальные файлы кластера и в репозиторий не входят. Примеры с ними требуют отдельных запускаторов. Доступный Python runner — [`ce_distill_tiger_yambda.py`](ce_distill_tiger_yambda.py); используйте `--help` для списка параметров.

Runner: `scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py`.
Для парного сравнения обычной выборки из q и curriculum с поправкой q/proposal:
`bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh --compare-q`.
Для 50M сохраняется last128; для 500M fine-tuning всегда использует случайную
подвыборку targets, по умолчанию sampled50. K=3 относится к кандидатам CE,
а не к размеру обучающей подвыборки. [Формулы и параметры CE](README.md#сравнение-двух-несмещённых-оценок-ce-по-q).
Отдельный эксперимент из начальных checkpoint TIGER exp2, с теми же
компонентами награды и калибровкой CE, что в [Amazon-эксперименте](README.md).

## Сокращённая сетка для 500M

Для 500M доступен отдельный запуск с сеткой наград и параметров,
как у [GPT-2 Yambda](README_GPT2_YAMBDA.md):

```bash
conda activate flowrec
bash scripts/scheduler/submit_ce_distill_tiger_yambda500_compact.sh
```

Он строит совместимые sampled50 caches один раз и создаёт **48 training tasks**:
4 набора наград (`sasrec`, `interaction`, hierarchical `similarity`,
произведение трёх) × 3 степени (`1`, `0.25`, `16`) × 2 rho (`1`, `0.1`)
× 2 режима (`reward`, `curriculum_smooth_is`), seed 42.
На каждый режим приходится массив `0-23`; K=3, плавный переход 5 эпох,
примесь q=0.1. В smooth применяется поправка q/m.

Для TIGER 500M выбирается до 50 случайных targets на пользователя из пула
последних 128; subset seed 42 фиксирован между эпохами, режимами CE и
конфигурациями. Контекст каждого выбранного target содержит до 50 items.
Размер подвыборки настраивается через `--train-random-n` (от 1 до 127), seed —
через `--train-subset-seed`. Значения 0 и 128 или больше запрещены, чтобы
fine-tuning не мог незаметно перейти на весь last128. Эти параметры не
меняют обработку 50M. Для каждого значения sampled N, subset seed и режима CE
используются соответствующие каталоги результатов и совместимые factor caches;
полные last128 checkpoints и caches для sampled50 не используются.

Проверить команды без подачи:

```bash
bash scripts/scheduler/submit_ce_distill_tiger_yambda500_compact.sh --dry-run
```

`--skip-cache` добавляйте только при наличии совместимых sampled caches.
По умолчанию оба массива обучения ждут одни и те же задачи подготовки
кешей. Настройки передаются после `--`, например:

```bash
bash scripts/scheduler/submit_ce_distill_tiger_yambda500_compact.sh -- \
  --train-random-n 30 --train-subset-seed 42
```

Смена N или subset seed создаёт отдельные кеши и запуски. Resume продолжает
только checkpoint с тем же протоколом и подвыборкой.

Для сводки именно по сокращённой сетке укажите её явно:

```bash
for sampling in reward curriculum_smooth_is; do
  python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py \
    --aggregate-only --datasets yambda_500m_listens \
    --combos sasrec interaction similarity sasrec_x_interaction_x_similarity \
    --powers 1 0.25 16 --rho-targets 1 0.1 --sampling "$sampling"
done
```

## Данные и сетка

Оба датасета используют стандартный `prefix_target` layout exp2:
`train_target_policy=last_n`, `train_last_n=128`, `max_history_len=50`.
Это исходный пул до 128 последних **обучающих targets** на пользователя;
для каждого target история содержит до 50 предыдущих положительных событий.
50M использует весь пул, 500M — фиксированную случайную подвыборку до N targets
на пользователя, по умолчанию N=50. Отбор выполняется без возвращения;
если targets меньше N, сохраняются все. Validation и test остаются отдельными
leave-one-out событиями и не сокращаются из-за подвыборки train.

Положительные события — `listen` с `played_ratio_pct >= 50`. Как в GFlow,
фильтр выражен через `positive_criterion`: остальные события доступны
interaction reward, но не становятся положительными обучающими targets.
Сохраняются ограничения и временное окно конфигурации exp2. Ожидаемый масштаб
исходных assets: 993 926 / 10 004 891 примеров в train-пуле и 112 943 / 374 283
items для 50M / 500M соответственно. Число выбранных
train-примеров sampled50 сохраняется в manifest и result.json; оно зависит
от числа доступных targets у каждого пользователя. Полный пул строится для
подготовки и проверки данных, но в обучение 500M целиком не передаётся.

По умолчанию **112 запусков**, по 56 на датасет:

| Параметр | Значения |
|---|---|
| Rewards | `sasrec`, `interaction`, `similarity`, все три пары, произведение трёх |
| Power | `0.25`, `1`, `4`, `16` |
| Rho, целевое отношение норм градиентов CE/SFT | `1`, `0.1` |
| Seed | `42` |
| Семплирование | `curriculum_smooth`, 3 выборки на историю |
| Длительность плавного перехода | 5 эпох |

`similarity` — hierarchical similarity по RQ-VAE. `sasrec` использует учителя
из соответствующего matched-протокола: обычный exp2 SASRec для 50M и
SASRec-SCE для 500M. Его transform — sigmoid с temperature 1.
Степень применяется ко всему произведению выбранных наград, с epsilon
floors exp4; это перебор общей степени, не независимых степеней компонентов.

## Плавное семплирование

Для каждой истории вычисляется нормированная награда полного каталога
`q(i|h) = R(h,i) / sum_j R(h,j)`. Каталог сортируется по убыванию `q` и
делится пополам по числу items; ties сохраняют порядок каталога.
Верхняя половина содержит `floor(N/2)` items, нижняя — остальные.

```text
p_t = (1 - alpha_t) * q_low + alpha_t * q_high
alpha_t = clip(global_microstep / transition_steps, 0, 1)
transition_steps = ceil(curriculum_epochs * batches_per_epoch)
```

Здесь `q_low` и `q_high` нормированы отдельно внутри своих половин.
В начале выбирается только нижняя половина; за 5 эпох масса верхней
плавно растёт до 1. После перехода выборки идут из верхней половины.
`--curriculum-epochs 0` сразу включает верхнюю половину.
Внутри выбранной половины вероятность пропорциональна `q`.

На каждую историю берутся **3 IID выборки с возвращением**, и усредняется
их `-log pi(i|h)`. Это CE текущей смеси `p_t`, без importance weights.
Items из истории и положительный target не исключаются, отдельного
принудительного добавления target в CE нет. Target используется в SFT.
Уменьшение числа выборок ускоряет декодирование TIGER; полный расчёт `q`
и сортировка по каталогу всё равно выполняются.

Итоговый loss — `L_SFT + lambda_D * CE`. `lambda_D` один раз калибруется
по нормам градиентов на 5 батчах. Batch 256 × accumulation 8, AdamW
lr 3e-5, weight decay 0.01. Validation каждую эпоху, NDCG@5 с beam 5;
для 500M фиксированная выборка 25% validation (seed 0), для 50M весь split.
Начальный checkpoint участвует в выборе best. Patience 5 после первых
10 эпох, максимум 500. Финальные validation/test полные: beam 20,
HR/NDCG/Coverage/ARP при K=5,10,20.

## Запуск

На кластере из корня репозитория:

```bash
conda activate flowrec
bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh
```

Вместо активации можно задать `PYTHON_BIN=/path/to/flowrec/bin/python`.
Вычислительные задачи используют этот интерпретатор; без `PYTHON_BIN`
активируют `CONDA_ENV`, по умолчанию `flowrec`.

Посмотреть команды без подачи:

```bash
bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh --dry-run
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py --list
```

Изменить длительность перехода, seeds или сузить сетку:

```bash
bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh -- \
  --curriculum-epochs 10 --seeds 42 43 44
bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh -- \
  --datasets yambda_500m_listens --combos sasrec_x_interaction_x_similarity \
  --powers 1 4 --rho-targets 0.1
```

Для готовых совместимых factor caches добавьте `--skip-cache` перед `--`.
Повторная подача продолжает незавершённые запуски; completed пропускаются.
`last.pt` сохраняется каждые 10 optimizer updates и в конце эпохи.
Прогресс curriculum и RNG восстанавливаются вместе с обучением.

## Ресурсы scheduler

Подготовка SASRec factors использует GPU. Interaction и hierarchical
similarity factors строятся на CPU. Обучение использует одну GPU на задачу.
Учтены [правила кластера](https://cluster.example.org/llms.txt) и
[характеристики оборудования](https://cluster.example.org/hardware/hpc-cluster).
Partition `rocky`; свой проектный аккаунт укажите через `ACCOUNT`.
`--mem` и номера узлов не используются.

| Этап | Yambda 50M | Yambda 500M |
|---|---|---|
| SASRec cache | V100 32 GB, `type_a\|type_c` | V100 32 GB, `type_b` |
| Interaction/similarity cache | CPU, `type_d`, 0 GPU | CPU, `type_b`, 0 GPU |
| Обучение | V100 32 GB, `type_a\|type_c` | A100 80 GB, `type_e` |
| Одновременные задачи каждого массива | 4 | 1 |

При batch 256 одна FP32-матрица по каталогу занимает примерно 110 / 366 MiB;
нужны также buffers сортировки, смеси, encoder и градиентов. Для 50M оценка
укладывается в V100. Для 500M выбран A100 80 GB: предыдущий full-last128 CE
с тем же batch успешно отработал на A100. Оценка 1.3 TB host RAM
не подтверждена измерением и не используется как основание для H200-only.
Подвыборка применяется после построения исходного пула, поэтому уменьшение
train не означает такое же уменьшение пикового потребления RAM подготовки.

У 500M CPU-cache array ждёт GPU-cache array; внутри CPU array компоненты
идут по одному. Обучение ждёт оба. У 50M cache arrays независимы. Лимит
параллелизма уменьшает конкуренцию этих задач за RAM, но не резервирует
всю память узла. Тип GPU можно изменить через `TRAIN_500M_CONSTRAINT`.
Фактическую полезную загрузку GPU и peak RAM следует проверить на первом
запуске, как требуют правила кластера.

Лимиты времени: обучение 50M — 3 дня, sampled50 500M — 24 часа; cache tasks
50M — 2 дня, 500M — 8 часов для SASRec и 16 часов для CPU-наград. Лимиты
увеличены относительно sampled15 с учётом большего числа train-примеров.
24 часа — начальный лимит с запасом для smooth: предыдущий full-last128
запуск занял 27.8 часа fit; точное время sampled50 станет известно после запуска. Лимит влияет на возможность размещения
задачи в расписании, а конкретные веса приоритета задаёт кластер:
[scheduler multifactor priority](https://scheduler.schedmd.com/priority_multifactor.html#general).
Настройки: `TRAIN_TIME` (общее переопределение), `TRAIN_500M_TIME`,
`CACHE_GPU_TIME`, `CACHE_CPU_TIME` (общие переопределения),
`CACHE_GPU_500M_TIME`, `CACHE_CPU_500M_TIME`, `MAX_PARALLEL_50M`,
`MAX_PARALLEL_500M`, `TRAIN_WORKERS`, `CACHE_GPU_WORKERS`, `CACHE_CPU_WORKERS`;
полный список — `bash scripts/scheduler/submit_ce_distill_tiger_yambda.sh --help`.

## Кеши и результаты

Новые кеши:

```text
artifacts/exp7-ce-distillation-tiger-yambda/yambda_50m_listens/last128/reward_factors/<component>/
artifacts/exp7-ce-distillation-tiger-yambda/yambda_500m_listens/sampled50_seed42/reward_factors/<component>/
```

Совместимые кеши предыдущего GFlow могут переиспользоваться после проверки
протокола: full last128 для 50M, только соответствующий sampled subset для 500M.
Автоматическое переиспользование требует готовых factors всех трёх компонентов
и доступных исходных config JSON; иначе выбирается отдельный каталог exp7.
Проверяются конфигурация данных, subset, число строк, метаданные targets
и порядок каталога. Кеши full-last128 и last8 для 500M не подходят; совпадения
одного числа строк недостаточно — должны совпасть N, subset seed и targets.
`--cache-root` задаёт другой родительский каталог; параметры power/rho/seed
не требуют отдельных factor caches.

Результаты каждого запуска:

```text
experiments/exp7-ce-distillation-tiger-yambda/yambda_50m_listens/<combo>/<ce_mode>/pow_<p>/rho_<r>/seed_<s>/
experiments/exp7-ce-distillation-tiger-yambda/yambda_500m_listens/sampled50_seed42/<combo>/<ce_mode>/pow_<p>/rho_<r>/seed_<s>/
```

Для произведения трёх наград сегмент `<combo>/` опущен, как в TIGER Amazon.
Внутри: `identity.json`, `resolved_config.json`, `data_reward_config.json`,
`calibration.json`, `training_log.jsonl`, `training_history.json`,
`checkpoints/{best,last}.pt`, `result.json`. Метрики и веса находятся в
`experiments/`; `artifacts/` содержит кеши наград.

Собрать сводку после завершения массивов:

```bash
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py --aggregate-only
```

`metrics.csv` и `summary.json` появятся в
ветке сводки, указанной runner; при включённом 500M она дополнительно
разделяется по N и seed подвыборки. Старые last128 результаты остаются на диске
и не считаются завершёнными sampled50 запусками.
Если меняли сетку, число targets, subset seed, CE samples или curriculum, передайте те же флаги
при агрегации. Power/rho выбираются отдельно для каждой комбинации наград
по полному validation NDCG@5, не по test.

`--list`, `--count`, `--aggregate-only` не загружают ML и datasets.
`--smoke` ограничивает число train/eval батчей и изолирует результаты,
но **всё равно строит полный исходный датасет**. Это не маленькая локальная
проверка для Yambda 500M; его запускать только на подходящем compute-узле.
