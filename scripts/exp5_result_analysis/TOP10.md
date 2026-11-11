# Test top-10 после первого и второго SFT-only

`export_top10.py` выгружает рекомендации GPT-2 и TIGER для всех тестовых примеров
Beauty 2014 (`beauty`) и Yambda 50M (`yambda_50m_listens`). Отдельно сохраняются
частоты items, рассчитанные только по train.

## Какие модели входят

Для каждой пары модель/датасет:

1. `sft`: **best.pt всех завершённых запусков/seeds первого SFT-only** выбранного семейства.
2. `sft2`: **best.pt всех завершённых запусков второго SFT-only** из `exp7-from-sft`.
3. `gflow` и `ce`: выбранные конфигурации дообучения **от первого SFT-only**,
   отдельно для каждого метода: максимум NDCG@10, максимум Coverage@10 и Pareto.
   Сохраняются best.pt всех seeds выбранной конфигурации.

Выбранные семейства первого SFT:

| Модель / датасет | Путь относительно репозитория (N — seed) | source_id второго этапа |
|---|---|---|
| GPT-2 / Beauty 2014 | `experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/beauty/sft_only/seed_N/checkpoints/best.pt` | `gpt2_beauty` |
| GPT-2 / Yambda 50M | `experiments/exp4-gflow-gpt2-yambda50-trial36-k3/yambda_50m_listens/sft_only/seed_N/checkpoints/best.pt` | `gpt2_yambda50_trial36` |
| TIGER / Beauty 2014 | `checkpoints/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05/seed_N/best.pt` | `tiger_beauty` |
| TIGER / Yambda 50M | `checkpoints/exp2-tuned_params/finetuned_matched/tiger/yambda_50m_listens/sft_only_lr3e-05/seed_N/best.pt` | `tiger_yambda50_matched` |

GPT-2 / Yambda проверяет происхождение первого SFT от
`checkpoints/exp2-tuned_params/gpt2_tiger_matched/yambda_50m_listens/trials/trial_36/best.pt`.
Вторые SFT, GFlow и CE читаются из канонических записей
`experiments/exp8-from-sft/<source_id>/sft_seed_N/jobs/*/result.json`.
Также учитывается GFlow TIGER Beauty из
`experiments/exp4-gflow-tiger-from-sft/beauty/sft_only_lr3e-05/sft_seed_N/`.
Его конфигурации остаются отдельными от exp8, в том числе при другом K.
Происхождение проверяется по сохранённым метаданным и source identity.

«Все чекпоинты» означает **только best каждого завершённого запуска/seeds**
первого и второго SFT. Это поведение по умолчанию; last.pt в этот экспорт не входят.
`--checkpoint-kinds best last` добавляет существующие last.pt первого и второго SFT.
GFlow/CE всегда используют best.pt, которому соответствуют метрики выбора.
Исходные пути весов сохраняются; отсутствующие веса не заменяются primary-копиями.

## Выбор конфигураций

Выбор по умолчанию использует **test** NDCG@10 и Coverage@10.
Генерация рекомендаций также идёт по всему test. Это описательный выбор по test.
При необходимости отдельного эксперимента `--selection-split validation`
переключает только метрики выбора конфигураций.
Метрики конфигурации усредняются по сохранённым seeds. Pareto — недоминируемый
фронт, min-max нормализация обеих метрик по фронту, затем минимум евклидова
расстояния до `(1, 1)` с равными весами. Постоянная координата считается равной 1.
При равенстве приоритет имеют NDCG, coverage, затем идентификатор конфигурации.
Если несколько критериев выбирают одну конфигурацию, инференс выполняется один
раз; все метки остаются в манифесте. Поэтому разных конфигураций может быть меньше трёх.

Учитываются завершённые запуски с полной оценкой и обеими нужными метриками.
Незавершённый запуск с одним файлом весов не становится кандидатом автоматически.
Манифест содержит критерии выбора, исходные метрики, параметры и пути каждого запуска.
Отсутствующие семейства перечисляются в `missing_groups`; перед вычислениями
проверка требует полный набор четырёх стадий для всех четырёх пар модель/датасет.

## Запуск

Из корня репозитория с доступными данными, весами и зависимостями:

```bash
OUT=artifacts/exp6_result_analysis/top10
python scripts/exp5_result_analysis/export_top10.py --mode discover --output-dir "$OUT"
python scripts/exp5_result_analysis/export_top10.py --mode check --output-dir "$OUT"
python scripts/exp5_result_analysis/export_top10.py --mode prepare --output-dir "$OUT" --cpu-threads 8
python scripts/exp5_result_analysis/export_top10.py \
  --mode generate \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 64 \
  --num-beams 20
python scripts/exp5_result_analysis/export_top10.py --mode verify --output-dir "$OUT"
```

Discovery фиксирует `manifest.json` в каталоге результатов. Для отдельного
манифеста передайте одинаковый `--manifest` всем этапам. Изменение отбора
моделей или пользователей требует нового манифеста. `--selection-split test`
и `--checkpoint-kinds best` — настройки по умолчанию.

Discovery читает JSON и требует NumPy; веса и содержимое сырых датасетов не
загружаются. `check` проверяет исходные файлы и полноту семейств.
Подготовка выполняется на CPU и заканчивается до генерации; генерация читает
готовые тестовые тензоры. Без индексов этапы выполняются последовательно.
`--prepare-index N` и `--run-index N` позволяют распределить независимые
протоколы и чекпоинты между процессами.

Для smoke укажите отдельный `--output-dir` и `--max-users 64` при discovery.
Подготовка всё равно строит исходные splits целиком. Пользователи выбираются
по hash raw ID с фиксированным seed, одинаково для совместимых протоколов.
Полный test выгружается без `--max-users`.

## Результаты и проверка

В `artifacts/exp6_result_analysis/top10/`:

- `recommendations/<run_id>.csv.gz` — отдельный файл для каждого чекпоинта,
  одна строка на тестовый пример:
  `example_index,user_id,target_item_id,item_1,...,item_10`.
  `item_1` — первый ранг; все ID исходные строковые.
- `recommendations/<run_id>.csv.json` — путь весов, стадия, seed, метки выбора,
  параметры генерации, число примеров, контрольные суммы и путь к популярности.
- `popularity/<preparation_id>.csv` — `item_id,train_count,train_fraction`,
  включая items с нулевой train-популярностью.
- `prepared/` — общие тестовые тензоры; `cache/` — выдачи NPZ для проверяемого повторного запуска.

`train_count` — число исходных положительных train-взаимодействий после
препроцессинга, до размножения prefix targets. Validation/test не учитываются.
`train_fraction = train_count / sum(train_count)`. Для разных исходных протоколов
используются отдельные файлы популярности; метаданные выдачи указывают нужный файл.

Генерация ограничена semantic IDs каталога. Уже просмотренные items разрешены,
как в исходной оценке. Каждый пример обязан получить ровно 10 уникальных items;
некорректная или неполная выдача вызывает ошибку.

После завершения генерации:

```bash
python scripts/exp5_result_analysis/export_top10.py --mode verify \
  --manifest artifacts/exp6_result_analysis/top10/manifest.json
```

Для нестандартного каталога результатов добавьте тот же `--output-dir`.
Проверка сверяет полноту, актуальные тестовые входы, параметры подвыборки,
метаданные и контрольные суммы. Повторный запуск использует только совместимые
кеши. `--force` перестраивает выбранный этап.

Ручные этапы: `--mode discover`, `check`, `prepare --prepare-index N`,
`generate --run-index N --device cuda`, `verify`. Без индекса prepare/generate
обрабатывает все записи последовательно.
