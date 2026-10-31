# Популярность айтемов и длина истории: init → SFT-only / GFlow

> **Доступность:** упомянутые ниже scheduler-запускаторы — локальные файлы кластера, в репозиторий они не входят. Их примеры требуют отдельно предоставленных запускаторов. Доступная точка входа — [`finetune_buckets.py`](finetune_buckets.py); параметры перечислены в `python scripts/exp6_result_analysis/finetune_buckets.py --help`.

`finetune_buckets.py` сравнивает сохранённые **best checkpoints** эксперимента 4
на одних и тех же тестовых пользователях. Обучение не требуется. Это анализ
изменений между init и результатом файнтюнинга; временная траектория по эпохам
не измеряется.

## Выбранные конфигурации

Для анализа используются выбранные конфигурации leave-one-out, перечисленные
ниже. Их параметры сохранены в `bucket_selection_provenance.json`.

| Данные / модель | best NDCG | best coverage | Pareto |
|---|---|---|---|
| Beauty / GPT-2 | sasrec, pow=4, DB, rho=1 | sasrec×interaction, pow=16, TB, rho=0.1 | тот же checkpoint, что best NDCG |
| Instruments / GPT-2 | interaction×sim_hier, pow=1, DB, rho=1 | interaction×sim_hier, pow=16, TB, rho=0.1 | sim_hier, pow=16, SubTB, rho=1 |
| Yambda 50M matched / TIGER | sasrec, pow=1, DB, rho=1 | sasrec×interaction×similarity, pow=16, SubTB, rho=1 | sasrec, pow=0.25, SubTB, rho=1 |
| Yambda 500M sampled-15 / TIGER | sasrec, pow=4, DB, rho=1 | sasrec×interaction, pow=16, SubTB, rho=1 | sasrec×similarity, pow=0.25, DB, rho=1 |

Во всех GFlow выбранных строках seed=42, число негативов=10, веса компонент=1.
У Beauty три метки соответствуют двум уникальным конфигурациям: обе метки
сохраняются в manifest, выдача дублирующего checkpoint генерируется один раз.
**19 запусков инференса**: 4 init + 4 основных SFT-only + 11 GFlow.
`--seeds 42 43 44` выбирает те же основные SFT-модели, каждая учитывается
один раз с фактическим seed из `provenance.json`. Межseedовый разброс GFlow
при этом не оценивается; общий init не считается независимым повтором.

Пути зафиксированы в `bucket_manifest.py`, доступны через `--mode discover`:

- GPT-2: `experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/<dataset>/`.
  SFT-only берётся из этой же семьи; отдельный dense-control не подставляется.
- TIGER 50M GFlow: `experiments/exp4-gflow-yambda50m-matched/`;
  SFT: `experiments/exp2-sft-finetune-matched/yambda_50m_listens/`.
- TIGER 500M GFlow: `experiments/exp4-gflow-yambda500m-matched-r15/`;
  SFT: `experiments/exp2-sft-finetune-matched-r15/yambda_500m_listens/`.

Missing weights остаются в manifest: локальное отсутствие не исключает
конфигурацию из эксперимента. Пути относительны корню репозитория, старые
абсолютные пути внутри captured configs разрешаются при загрузке. Границы
каталога, semantic IDs, исходный checkpoint и matched/random15-протокол
проверяются по имеющимся артефактам. Выбранный в таблице test результат —
основание этого описательного анализа, а не независимого подбора конфигурации.

## Разбиение и метрики

Популярность — число **тренировочных взаимодействий** из существующей
`training_item_popularity`, без val/test и без повторного подсчёта prefix-таргетов.
Все catalog items получают один фиксированный бакет:

- `unseen`: training count=0;
- `tail`: средний процентильный ранг среди items с положительной частотой ≤50%;
- `middle`: (50%, 80%];
- `head`: >80%.

Равные частоты не разделяются; поэтому размеры отличаются от номинальных
50/30/20%. `catalog_buckets.csv.gz` содержит каждый item, частоту и группу.

Для пользователей используются интервалы полной истории `0–5`, `6–10`,
`11–20`, `21–50`, `51–100`, `101–200`, `201+` (меняются через
`--history-edges`). Длина берётся из `target_position`: положительные события
до test target после исходной предобработки, до отсечения контекста модели
и удаления холодных items из истории. В LOO сюда входит validation item.
Сохраняются также длина фактического контекста и SHA256 его содержимого.
Группы пользователей и пересечения «популярность таргета × длина истории»
одинаковы для всех моделей. Пустые группы остаются с `num_users=0` и NaN.

При K=5/10/20 рассчитываются:

- **Качество по группе таргета:** HR/Recall (один таргет) и NDCG на исходных
  позициях top-K. Рекомендации не фильтруются по бакету и не перенумеровываются.
- **Выдачи пользователей группы:** coverage всего каталога, ARP, Gini показов,
  entropy и exp(entropy), среднее пересечение выдач разных пользователей.
- **Представленность каждого бакета в выдаче:** доля показов, уникальные items,
  coverage с размером именно этого бакета в знаменателе, концентрация показов.
- **Парные изменения:** GFlow против init и SFT того же seed; SFT против init.
  Число новых/потерянных попаданий, улучшенных/ухудшенных NDCG, изменения ранга,
  пересечение выдач и Δметрик. `accuracy_attribution.csv` даёт вклады бакетов
  в общий ΔHR/ΔNDCG: их сумма внутри одного `cohort_type` равна общему изменению.
- **Конкретные items:** число test targets, попадания, HR/NDCG, показы до/после,
  новые/исчезнувшие из охвата items, выигрыши/потери и вклад в общую accuracy.

Coverage и концентрация показов характеризуют **агрегатное разнообразие**;
семантическое разнообразие внутри списка и энтропия вероятностей модели здесь
не измеряются. Размеры пользовательских групп различаются: coverage между
разными группами зависит также от числа пользователей; сравнение моделей
внутри одной группы имеет общий размер выборки. Популярные test targets
получают больший вес в micro-метриках; per-item таблицы позволяют видеть это.

`--bootstrap-samples 500` добавляет парные интервалы ΔHR/ΔNDCG по пользователям
(по умолчанию 0). Это неопределённость тестовой выборки при фиксированных
моделях; для малых групп интервалы могут быть неинформативны. Межseedовая
неопределённость обучения из этих интервалов не следует.

## Кластер: полный запуск

Из корня checkout на кластере, в активном окружении `flowrec`:

```bash
# Просмотр точных задач и зависимостей, без submit-worker и загрузки данных.
DRY_RUN=1 bash scripts/exp6_result_analysis/scheduler/submit.sh

# Поставить весь эксперимент в scheduler.
bash scripts/exp6_result_analysis/scheduler/submit.sh

# Старые значения SFT_SEEDS тоже выбирают те же основные модели:
SFT_SEEDS="42 43 44" BUCKET_OUTPUT=artifacts/exp6_result_analysis/buckets_sft3 \
  bash scripts/exp6_result_analysis/scheduler/submit.sh
```

Лаунчер создаёт зависимости **CPU prepare array → GPU inference array → CPU
analysis array**. В каждой GPU-задаче один checkpoint, поэтому конфигурации
и seeds могут выполняться параллельно. 500M имеет отдельную цепочку;
остальные датасеты не ждут его подготовки. Не запускайте полную подготовку
или анализ на login-узле.

Ресурсы выбраны по [правилам HSE](https://cluster.example.org/llms.txt) и
[характеристикам кластера](https://cluster.example.org/hardware/hpc-cluster):

- Prepare: только CPU, 8 потоков, zero-worker dataloader. Beauty, Instruments,
  50M — `type_d`; 500M — `type_b` с 1.5 TB RAM, **без запроса GPU**.
  Исходный datamodule всё ещё строит last128 train layout для точного каталога;
  random15 не гарантирует снижения пика подготовки. Большой RAM-класс выбран
  консервативно по существующему resource-профилю проекта, новый пик не измерен.
- Инференс: 1 **V100 32 GB** (`type_a|type_b|type_c`), 4 CPU, batch64, beam20, FP32.
  Загружаются только компактные eval tensors, semantic IDs и веса. Нет optimizer,
  reward-кэшей и raw Yambda; A100/H200 для этого расчёта не требуются по оценке
  размеров моделей и буферов. Фактическая скорость/память на GPU ещё требуют
  проверки первым запуском; при OOM уменьшите `BATCH_SIZE`.
- Анализ: только CPU (`type_d`), 2 потока. Для всех этапов укажите свой
  проектный аккаунт через `ACCOUNT`. `mem` и номера узлов не используются.
  `GPU_CONCURRENCY=4` на каждую цепочку.

Подготовка на CPU защищает от простоя GPU во время чтения большого Yambda.
После первого запуска проверьте реальную загрузку через средства кластера
и скорректируйте batch size / лимит времени. Задачи не создают искусственной
нагрузки. Ограничение параллельности относится к этому эксперименту; scheduler
без учёта RAM не гарантирует отсутствия чужой нагрузки на high-memory узле.

Переменные: `BUCKET_OUTPUT`, `DATASETS` (имена из таблицы manifest),
`SFT_SEEDS`, `ACCOUNT`, `CONDA_ENV`, `BATCH_SIZE`, `GPU_CONCURRENCY`,
`BOOTSTRAP_SAMPLES`. Для короткого первого прогона используйте отдельный
`BUCKET_OUTPUT` и `MAX_USERS=128`; даже такой smoke подготавливает исходные
splits целиком, поэтому большой Yambda требует CPU-задачи с тем же RAM-классом.

## Отдельные этапы и локальная проверка

```bash
python scripts/exp6_result_analysis/finetune_buckets.py --mode discover
python scripts/exp6_result_analysis/finetune_buckets.py --mode prepare --datasets beauty
python scripts/exp6_result_analysis/finetune_buckets.py --mode generate --run-index 0 --device cuda
python scripts/exp6_result_analysis/finetune_buckets.py --mode analyze --datasets beauty
```

Последняя команда требует **все** выбранные выдачи Beauty; отсутствие файла
завершает анализ ошибкой и не превращает полный эксперимент в частичный.
Для CPU smoke создайте отдельный manifest/каталог:

```bash
OUT=artifacts/exp6_result_analysis/buckets_smoke
python scripts/exp6_result_analysis/finetune_buckets.py --mode discover --datasets beauty --output-dir "$OUT"
python scripts/exp6_result_analysis/finetune_buckets.py --mode prepare --output-dir "$OUT" --max-users 8
python scripts/exp6_result_analysis/finetune_buckets.py --mode generate --output-dir "$OUT" --run-index 0 --device cpu --batch-size 4
```

Все относительные пути считаются от корня репозитория. Не меняйте manifest
во время активного array: индексы задач относятся к его фиксированному порядку.
При повторном запуске совместимые кэши проверяются и используются снова.
При изменениях inputs/protocol подготовка и генерация пересчитываются;
`--force` принудительно обновляет этап. NPZ, prepared tensors, catalog и
sidecar проверяются content-hash, исходные большие parquet — size/mtime.
FP32 и текущая версия декодера могут отличаться от старой оценки: расхождения
с исходными метриками полного test не скрываются в `metric_validation.csv`.

## Артефакты

`artifacts/exp6_result_analysis/finetune_buckets/<dataset>/`:

- `report.html` — автономный отчёт с встроенными графиками;
- `figures/*.png`, `*.svg` — NDCG и ΔNDCG по бакетам, экспозиция и охват,
  матрицы изменений по пересечению групп;
- `tables/summary.csv`, `cohorts.csv`, `popularity_exposure.csv` — метрики;
- `tables/comparisons.csv`, `accuracy_attribution.csv` — парные изменения;
- `tables/items.csv.gz`, `item_deltas.csv.gz`, `catalog_buckets.csv.gz` — items;
- `tables/definitions.csv`, `metric_validation.csv`, `analysis_metadata.json`
  — определения, сверка и протокол;
- `prepared/` — один набор frozen eval inputs на датасет;
- `cache/<run_id>.npz` и `.json` — выдачи и provenance каждого checkpoint.

```bash
python -m pytest tests/test_bucket_manifest.py tests/test_bucket_metrics.py \
  tests/test_bucket_inference.py tests/test_finetune_buckets.py -q
```
