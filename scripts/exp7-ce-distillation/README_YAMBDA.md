# TIGER: CE distillation на Yambda 50M и 500M

Runner: `scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py`.
Для парного сравнения обычной выборки из q и curriculum с поправкой q/proposal:
`bash scripts/slurm/submit_ce_distill_q_comparison.sh --profile tiger-yambda --skip-cache`.
Сохраняются last128 и K=3; [формулы, сетка и параметры](README.md#сравнение-двух-несмещённых-оценок-ce-по-q).
Отдельный эксперимент из начальных checkpoint TIGER exp2, с теми же
компонентами награды и калибровкой CE, что в [Amazon-эксперименте](README.md).

## Данные и сетка

Оба датасета используют стандартный `prefix_target` layout exp2:
`train_target_policy=last_n`, `train_last_n=128`, `max_history_len=50`.
Это до 128 последних **обучающих targets** на пользователя; для каждого
target история содержит до 50 предыдущих положительных событий. Validation
и test остаются отдельными leave-one-out событиями. Дополнительного отбора
последних 8 или случайных 15 targets нет.

Положительные события — `listen` с `played_ratio_pct >= 50`. Как в GFlow,
фильтр выражен через `positive_criterion`: остальные события доступны
interaction reward, но не становятся положительными обучающими targets.
Сохраняются ограничения и временное окно конфигурации exp2. Ожидаемый масштаб
исходных assets: 993 926 / 10 004 891 train-примеров и 112 943 / 374 283 items
для 50M / 500M соответственно.

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
Внутри выбранной половины вероятность по-прежнему пропорциональна `q`.

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
bash scripts/slurm/submit_ce_distill_tiger_yambda.sh
```

Вместо активации можно задать `PYTHON_BIN=/path/to/flowrec/bin/python`.
Вычислительные задачи используют этот интерпретатор; без `PYTHON_BIN`
активируют `CONDA_ENV`, по умолчанию `flowrec`.

Посмотреть команды без подачи:

```bash
bash scripts/slurm/submit_ce_distill_tiger_yambda.sh --dry-run
python scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py --list
```

Изменить длительность перехода, seeds или сузить сетку:

```bash
bash scripts/slurm/submit_ce_distill_tiger_yambda.sh -- \
  --curriculum-epochs 10 --seeds 42 43 44
bash scripts/slurm/submit_ce_distill_tiger_yambda.sh -- \
  --datasets yambda_500m_listens --combos sasrec_x_interaction_x_similarity \
  --powers 1 4 --rho-targets 0.1
```

Для готовых совместимых factor caches добавьте `--skip-cache` перед `--`.
Повторная подача продолжает незавершённые запуски; completed пропускаются.
`last.pt` сохраняется каждые 10 optimizer updates и в конце эпохи.
Прогресс curriculum и RNG восстанавливаются вместе с обучением.

## Ресурсы Slurm

Подготовка SASRec factors использует GPU. Interaction и hierarchical
similarity factors строятся на CPU. Обучение использует одну GPU на задачу.
Учтены [правила кластера](https://hpc.hse.ru/llms.txt) и
[характеристики оборудования](https://hpc.hse.ru/hardware/hpc-cluster).
Partition `rocky`, account `proj_1752`; без `--mem` и номеров узлов.

| Этап | Yambda 50M | Yambda 500M |
|---|---|---|
| SASRec cache | V100 32 GB, `type_a\|type_c` | V100 32 GB, `type_b` |
| Interaction/similarity cache | CPU, `type_d`, 0 GPU | CPU, `type_b`, 0 GPU |
| Обучение | V100 32 GB, `type_a\|type_c` | H200 141 GB, `type_h` |
| Одновременные задачи каждого массива | 4 | 1 |

При batch 256 одна FP32-матрица по каталогу занимает примерно 110 / 366 MiB;
нужны также buffers сортировки, смеси, encoder и градиентов. Для 50M оценка
укладывается в V100. Выбор H200 для 500M обусловлен прежде всего RAM узла
(от 1.5 TB): в репозитории зафиксирован старый пик 1.3 TB при построении
датасета, который после оптимизации временного окна не перемеряли.
Это консервативная оценка, не измерение нового CE-запуска.

У 500M CPU-cache array ждёт GPU-cache array; внутри CPU array компоненты
идут по одному. Обучение ждёт оба. У 50M cache arrays независимы. Лимит
параллелизма уменьшает конкуренцию этих задач за RAM, но не резервирует
всю память узла. После измерения нового запуска тип GPU можно изменить
через `TRAIN_500M_CONSTRAINT`, например на `type_e`, если RAM достаточно.
Фактическую полезную загрузку GPU и peak RAM следует проверить на первом
запуске, как требуют правила кластера.

Лимиты времени: обучение 3 дня, cache tasks 2 дня. Настройки:
`TRAIN_TIME`, `CACHE_GPU_TIME`, `CACHE_CPU_TIME`, `MAX_PARALLEL_50M`,
`MAX_PARALLEL_500M`, `TRAIN_WORKERS`, `CACHE_GPU_WORKERS`, `CACHE_CPU_WORKERS`;
полный список — `bash scripts/slurm/submit_ce_distill_tiger_yambda.sh --help`.

## Кеши и результаты

Новые кеши:

```text
artifacts/exp7-ce-distillation-tiger-yambda/<dataset>/last128/reward_factors/<component>/
```

Совместимые полные кеши предыдущего GFlow могут переиспользоваться из
`artifacts/exp4-reward_ablation/gflow_yambda50m_matched/reward_factors/` и
`artifacts/exp4-reward_ablation/gflow_yambda500m_last128/reward_factors/`.
Автоматическое переиспользование требует готовых factors всех трёх компонентов
и доступных исходных config JSON; иначе выбирается отдельный каталог exp7.
Проверяются конфигурация данных, subset, число строк, метаданные targets
и порядок каталога. Кеши `last8` и `matched_r15` не подходят.
`--cache-root` задаёт другой родительский каталог; параметры power/rho/seed
не требуют отдельных factor caches.

Результаты каждого запуска:

```text
experiments/exp7-ce-distillation-tiger-yambda/<dataset>/<combo>/curriculum_smooth_n3_epochs5/pow_<p>/rho_<r>/seed_<s>/
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
`experiments/exp7-ce-distillation-tiger-yambda/ablations/curriculum_smooth_n3_epochs5/`.
Если меняли сетку, число samples или curriculum, передайте те же флаги
при агрегации. Power/rho выбираются отдельно для каждой комбинации наград
по полному validation NDCG@5, не по test.

`--list`, `--count`, `--aggregate-only` не загружают ML и datasets.
`--smoke` ограничивает число train/eval батчей и изолирует результаты,
но **всё равно строит полный исходный датасет**. Это не маленькая локальная
проверка для Yambda 500M; его запускать только на подходящем compute-узле.
