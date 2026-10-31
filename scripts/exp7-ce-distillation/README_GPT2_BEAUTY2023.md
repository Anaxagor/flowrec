# GPT-2: CE distillation на Amazon Beauty 2023

> **Доступность:** Slurm-запускаторы ниже — локальные файлы кластера и в репозиторий не входят. Примеры с ними требуют отдельных запускаторов. Доступный Python runner — [`ce_distill_gpt2_beauty2023.py`](ce_distill_gpt2_beauty2023.py); используйте `--help` для списка параметров.

`ce_distill_gpt2_beauty2023.py` дообучает тот же исходный GPT-2 checkpoint,
что `gflow_finetune_gpt2_amazon_fullhist.py`:

```text
checkpoints/exp2/loo_split/initial/gpt2/beauty_2023/best.pt
```

Сейчас это exp2 trial 4, модель с четырьмя слоями, шириной 512 и четырьмя
головами attention. Архитектура и dropout читаются из checkpoint и
`best_trial.json`; файлы модели входят в identity запуска.

## Сетка и функция потерь

По умолчанию **24 конфигурации** при seed 42:

- награды: `sasrec`, `interaction`, `similarity` (hierarchical),
  `sasrec_x_interaction_x_similarity`;
- степени: `0.25`, `1`, `16`, общая степень применяется к произведению;
- `rho`: `1`, `0.1`;
- `curriculum_smooth_is`, **3 CE-кандидата на историю**, как K=3 в GFlow.

Используется существующая реализация `RewardDistillationMethod`:

```text
loss = SFT + lambda_CE * mean[-q(i)/m(i) * log pi(i | history)]
i ~ m, с возвращением
m = epsilon*q + (1-epsilon)*[(1-alpha)*q_low + alpha*q_high]
epsilon = 0.1
alpha = min(1, micro_step / transition_steps)
```

Каталог сортируется по награде и делится на нижнюю и верхнюю половины.
`q_low` и `q_high` сохраняют пропорции исходного `q` внутри своих половин.
Переход от низких к высоким наградам занимает **5 эпох** по умолчанию
(`--curriculum-epochs`). Примесь полного `q` сохраняет поддержку всего каталога,
включая края перехода. Используются сырые веса `q/m`, без клиппинга и
самонормировки, поэтому целевой CE и его градиент остаются по полному `q`.
Из выборки не исключаются target и просмотренные items.

Награды, SASRec teacher, RQ-VAE geometry и epsilon floors совпадают с GFlow.
В частности, floor после степени сохранён и при `power=16`. Калибровка
`lambda_CE` по нормам градиентов использует обычную выборку из `q`, пять
батчей и заданный `rho`; коэффициент сохраняется для resume.

## Данные и обучение

Используется `configs/exp2-tuned-params/dataset_amazon_2023.yaml`: verified
purchases, leave-one-out и прежние фильтры. Обучающий layout — prefix,
полученный из активных меток исходных dense-causal окон длины 20 с шагом 20.
Сохраняются границы контекстов, каталог и исходные validation/test rows;
дополнительного tail/random subset нет. Неположительные события сохраняются
для interaction reward. Проверка dense/prefix сохраняется в
`dense_prefix_audit.json`.

В существующем GFlow запуске эта постановка даёт 3 493 253 обучающих примера,
163 050 items и по 662 540 примеров validation/test. CE проверяет реальные
числа и соответствие reward caches при запуске, не подставляя эти значения.

CE вызывает тот же builder, что GFlow. Регрессионные тесты сравнивают
конфигурацию входных данных,
фильтров и токенизации, а на небольшом датасете — каждую строку train/val/test,
её токенизированное представление, каталог и выбранные индексы. Отдельно
проверяется совпадение prefix-примеров с активными метками causal-окон,
включая границы контекстов. Числа полного датасета подтверждены результатами
GFlow; при запуске CE соответствие проверяется заново.

- Batch **128 × accumulation 16 = 2048**; FP16, lr `3e-5`, weight decay `0.01`.
- Полная validation после каждой эпохи, beam 5, монитор NDCG@5; исходный
  checkpoint участвует в выборе лучшего. Patience 5, минимум 10 эпох,
  максимум 500, как в matched CE/GFlow протоколе.
- Финальная оценка лучшего checkpoint: полные validation/test, beam 20,
  HR/NDCG/Coverage/ARP при K=5,10,20.
- Resume включён; промежуточный checkpoint каждые 100 optimizer updates
  (`--checkpoint-every-updates`). Identity защищает от смены данных,
  гиперпараметров или reward factors в существующей директории.

## Запуск на кластере

Из корня проекта:

```bash
conda activate flowrec
bash scripts/slurm/submit_ce_distill_gpt2_beauty2023.sh
```

Вместо активации окружения можно задать `PYTHON_BIN` на Python окружения
flowrec. Helper создаёт GPU-cache array для SASRec, CPU-cache array для
interaction/hierarchical similarity и training array на 24 конфигурации,
который ждёт успешного завершения обоих cache arrays. Factors строятся один
раз на компонент, затем используются всеми степенями и значениями rho.
Проверенные существующие caches в выбранном cache-root переиспользуются.

Предпросмотр команд без отправки задач и без ML-зависимостей:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_beauty2023.sh --dry-run
python scripts/exp7-ce-distillation/ce_distill_gpt2_beauty2023.py --list
```

Повторный запуск с готовыми CE caches:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_beauty2023.sh --skip-cache
```

Для использования существующих **GFlow dense-prefix** factors:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_beauty2023.sh --skip-cache -- \
  --cache-root artifacts/exp4-reward_ablation/gflow_gpt2_amazon_fullhist_dense_prefix
```

Worker проверяет manifests, training context, subset и semantic catalog,
сверяет первые 256 targets с данными и все строки между компонентами.
Cache-root надо выбирать до первого запуска
конфигурации: переключение factors при resume меняет identity.

Параметры Python передаются после `--`, например:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_beauty2023.sh -- \
  --curriculum-epochs 10 --seeds 42 43 44
```

Это уже другая сетка/длительность curriculum; не изменяйте параметры
одновременно работающего массива. `--smoke` ограничивает обучение и caches,
но всё ещё подготавливает весь датасет; его также запускают на compute-узле.

## Ресурсы и результаты

Укажите свой проектный аккаунт через `ACCOUNT`; partition `rocky`. Обучение:
одна A100 80 GB или H100
80 GB (`type_e|type_f`) на задачу, как у GFlow Beauty 2023; лимит времени —
3 дня. SASRec factors: одна V100 32 GB; interaction/similarity factors:
CPU без GPU. Нет `--mem` и привязки к номерам узлов.
Учтены [правила HSE](https://hpc.hse.ru/llms.txt) и
[характеристики оборудования](https://hpc.hse.ru/hardware/hpc-cluster).

В GFlow на A100 для этой модели был измерен peak GPU около 10.94 GiB.
Это ориентир для выбора карты, не замер CE: CE дополнительно реконструирует
полное `q` и сортирует каталог. При batch 128 одна FP32-матрица
128 × 163 050 занимает около 80 MiB. Фактическую загрузку и peak RAM/VRAM
следует проверить после первого запуска.

```text
experiments/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/
  baseline/init_metrics.json
  [<single_reward>/]curriculum_smooth_is_n3_epochs5_qmix0p1/
    pow_<power>/rho_<rho>/seed_42/
      result.json
      training_history.json
      training_log.jsonl
      identity.json
      resolved_config.json
      calibration.json
      dense_prefix_audit.json
      checkpoints/{best,last}.pt
artifacts/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/reward_factors/
```

У произведения трёх наград нет дополнительной папки `<single_reward>`.
Сводку текущей сетки можно получить через `--aggregate-only`.
Общий `report_ce_results.py` включает этот CE root в итоговые таблицы.
