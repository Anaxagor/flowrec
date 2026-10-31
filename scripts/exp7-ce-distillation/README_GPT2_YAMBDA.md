# GPT-2: CE на Yambda 50M и 500M

> **Доступность:** Slurm-запускаторы ниже — локальные файлы кластера и в репозиторий не входят. Примеры с ними требуют отдельных запускаторов. Доступный Python runner — [`ce_distill_gpt2_yambda.py`](ce_distill_gpt2_yambda.py); используйте `--help` для списка параметров.

Runner: `scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py`.

## Yambda 500M, trial 4, sampled50

Отдельный запуск CE на фиксированных 50 случайных targets из последних 128
на пользователя, subset seed 42:

```bash
conda activate flowrec
bash scripts/slurm/submit_ce_distill_gpt2_yambda500_sampled50.sh
```

Wrapper запускает только Yambda 500M из исходного matched GPT-2 **trial 4**.
Сетка: `sasrec`, `interaction`, hierarchical `similarity`, произведение трёх;
степени `1`, `0.25`, `16`; rho `1`, `0.1`; seed 42. Всего **48 конфигураций**:
24 с обычной выборкой из q и 24 с `curriculum_smooth_is` (поправка q/m).
K=3 кандидата CE, плавный переход за 5 эпох, примесь полного q=0.1.

Сначала строятся общие для двух режимов кеши: SASRec на GPU, две остальные
награды на CPU. Обучение ждёт обе задачи подготовки. `--skip-cache` нужен
только при наличии готовых совместимых sampled50-кешей. План без отправки:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_yambda500_sampled50.sh --dry-run
```

Результаты, checkpoint и кеши sampled50 хранятся отдельно от sampled15:

```text
experiments/exp7-ce-distillation-gpt2-yambda/yambda_500m_listens/sampled50_seed42/
artifacts/exp7-ce-distillation-gpt2-yambda/sampled50_seed42/yambda_500m_listens/
```

В результатах находится также `cache_build/` с конфигурациями и аудитом
подготовки. В artifacts — `reward_factors/` и `source_assets/` с таблицей
перевода ID учителя. Обучение стартует с исходного trial 4; возобновление
возможно только из совместимого checkpoint этой же sampled50-конфигурации.
Sampled15-запуски являются самостоятельным экспериментом.

По умолчанию: одна A100 80 GB (`type_e`), 4 CPU, лимит обучения **48 часов**,
SASRec cache — 8 часов, CPU cache — 16 часов. Свой проектный аккаунт
укажите через `ACCOUNT` при вызове launcher.
Обоснование и ограничения оценки времени приведены ниже.

## Общий launcher

Запуск сетки для двух датасетов с кешами (50M last128, 500M sampled15):

```bash
conda activate flowrec
bash scripts/slurm/submit_ce_distill_gpt2_yambda.sh
```

Вместо активации можно задать `PYTHON_BIN=/path/to/flowrec/bin/python`.
Посмотреть план без отправки заданий:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_yambda.sh --dry-run
```

## Модели и данные

| Датасет | Исходный GPT-2 | Обучающие targets |
|---|---|---|
| `yambda_50m_listens` | matched exp2, trial 36 | Все последние 128 на пользователя, как в TIGER CE |
| `yambda_500m_listens`, общий launcher | matched exp2, trial 4 | До 15 случайных из последних 128 на пользователя; фиксированный subset seed 42 |
| `yambda_500m_listens`, отдельный sampled50 launcher | matched exp2, trial 4 | До 50 случайных из последних 128 на пользователя; фиксированный subset seed 42 |

Checkpoint берётся из
`checkpoints/exp2/loo_split/initial/gpt2/<dataset>/last_128/best.pt` (trial36 для 50M, trial4 для 500M).
Нужны также его `resolved_config.json` и сохранённый аудит данных
(в `resume_identity.data_audit` или `<dataset>/data_audit.json`).
Используется сохранённая архитектура trial; для trial 4 дополнительно
проверяется отсутствие NaN/Inf во всех весах. Состояние optimizer
исходного обучения не переносится.

Датасет строится с `prefix_target`, по записанному протоколу
`dense_causal_match_prefix` выбранного checkpoint. Этот протокол уже
согласован с полными prefix-историями: до 50 предыдущих положительных
событий на target. Проверяется совпадение targets, контекстов, каталога,
semantic IDs и held-out splits с аудитом исходного trial. Здесь
`train_prefix_from_dense=False`: оконное преобразование из старых Amazon
экспериментов не соответствует выбранным Yambda trial.

В 500M случайный отбор targets выполняется **после** построения и проверки
полного last128 pool. Отбор фиксирован между эпохами, режимами CE, степенями и rho;
он не укорачивает историю выбранного target и не меняет validation/test.
В общем runner размер задаётся через `--train-random-n` (1..127, default15),
seed — через `--train-subset-seed` (default42). Это параметры только 500M.
Отдельный sampled50 wrapper фиксирует N=50; другую подвыборку можно задать
через общий launcher. Если targets пользователя меньше N, сохраняются все.
Семплирование обучающих targets и 3 items из распределения наград — разные операции.

Положительные события и фильтры берутся из сохранённого trial.
Дополнительные события сохраняются для interaction reward. SASRec
использует того же учителя, что matched GFlow; перед построением его
кеша проверяется соответствие item IDs через исходные raw IDs.
Нужны teacher checkpoint, trial JSON и `training_summary.json`,
а для hierarchical similarity — соответствующий RQ-VAE asset.
Проверка каталога учителя может потребовать построения его датасета.

## Сетка и CE

По умолчанию **96 запусков**, по 24 на датасет и режим:

| Параметр | Значения |
|---|---|
| Награды | `sasrec`, `interaction`, `similarity`, `sasrec_x_interaction_x_similarity` |
| Общая степень награды | `1`, `0.25`, `16` |
| Rho | `1`, `0.1` |
| Семплирование | `reward`, `curriculum_smooth_is` |
| Выборки на историю | 3 IID с возвращением |
| Seed обучения | 42 |
| Плавный переход | 5 эпох |
| Примесь полного q | 0.1 |

`similarity` означает hierarchical similarity. Для тройки степень
применяется к произведению трёх компонент, с epsilon floors
exp4; отдельные степени компонент не перебираются.

В режиме `reward` items выбираются непосредственно из полного
`q(i|h) ∝ R(h,i)`. В `curriculum_smooth_is` каталог сортируется по убыванию
q и делится пополам по числу items. Каждая половина отдельно нормируется:

```text
alpha = clip(global_microstep / transition_steps, 0, 1)
c = (1 - alpha) * q_low + alpha * q_high
m = 0.9 * c + 0.1 * q
sampled CE = mean[-q(i) / m(i) * log pi(i|h)],  i ~ m
```

В начале масса смещена к низким наградам, за 5 эпох — к высоким.
Примесь полного q сохраняет поддержку; сырые веса q/m дают несмещённую
оценку CE по q. Веса не обрезаются и не нормируются по выборке.
Items истории и положительный target из q не исключаются.

Общий loss: `L_SFT + lambda_CE * CE`. Rho задаёт целевое отношение норм
градиентов CE/SFT при однократной калибровке lambda на пяти батчах.
Для обоих режимов калибровка использует обычную выборку из q.
Используются AdamW, lr `3e-5`, weight decay `0.01`, BF16 на GPU.
По умолчанию batch × accumulation: 128 × 16 для 50M, 256 × 8 для 500M.

Validation выполняется каждую эпоху, checkpoint выбирается по NDCG@5
с beam 5. Для 50M используется весь validation; для 500M — фиксированные
25%, seed 0. Начальный checkpoint участвует в выборе best; patience 5
после первых 10 эпох, максимум 500 эпох. Финальная оценка best использует
полные validation/test, beam 20, HR/NDCG/Coverage/ARP при K=5,10,20.

## Настройки запуска

Аргументы runner передаются после `--`:

```bash
# Только 500M, с переходом за 10 эпох.
bash scripts/slurm/submit_ce_distill_gpt2_yambda.sh -- \
  --datasets yambda_500m_listens --curriculum-epochs 10

# Повторный запуск с уже готовыми совместимыми кешами.
bash scripts/slurm/submit_ce_distill_gpt2_yambda.sh --skip-cache
```

Helper всегда создаёт оба режима на одной сетке. Незавершённые запуски
продолжаются с `last.pt`, завершённые пропускаются. Checkpoint сохраняется
каждые 10 optimizer updates и в конце эпохи; восстанавливаются optimizer,
RNG и прогресс curriculum. Совместимость продолжения проверяется по identity.
Изменение N или subset seed для 500M автоматически добавляет отдельную ветку
`sampledN_seedS` к run/cache-путям, кроме default15/42.

Для проверки сетки без Slurm, ML-пакетов и загрузки данных:

```bash
python scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py --list
python scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py --count
```

Runner выбирает один режим, поэтому `--count` выдаёт 48; helper запускает
его дважды. `--smoke` ограничивает обучение и изолирует outputs, но всё
равно строит полный исходный датасет; для 500M это задача compute-узла.

## Ресурсы и расположение результатов

Helper создаёт четыре training arrays: по одному на датасет/режим.
До них строятся общие кеши: SASRec на GPU, interaction и similarity
на CPU. Зависимости Slurm не позволяют начать обучение до готовности
кешей. Partition `rocky`; свой проектный аккаунт укажите через `ACCOUNT`.
`--mem` и выбор узла не используются.
Учтены [правила кластера](https://hpc.hse.ru/llms.txt).

Для 50M используется одна A100/H100 80 GB (`type_e|type_f`) на training
task, как в выбранном GFlow-профиле. Общий launcher использует для 500M
H200 (`type_h`). У отдельного sampled50 launcher выбран A100
(`type_e`): на нём успешно выполнен sampled50 SFT GPT-2 trial 4 с peak
allocated около 12.85 GiB. Для CE дополнительно нужны матрицы наград по
каталогу и декодирование K=3 кандидатов; 80 GB дают запас относительно SFT.
Это оценка пригодности, а не замер CE sampled50 на A100.
Полный last128 pool строится до подвыборки, поэтому пиковая RAM подготовки
не сокращается пропорционально N. Оценка 1.3 TB не подтверждена
измерением; фактический host RAM peak текущего CE неизвестен.
SASRec caches используют V100; CPU caches выделяют 0 GPU.

Лимиты параллелизма: 4 для 50M, 1 для 500M **на каждый массив**;
два режима могут выполняться одновременно. Cache-фазы 500M идут
последовательно. В общем launcher лимит training task — 3 дня, cache task — 2 дня.
Отдельный sampled50 launcher задаёт 48 часов train, 8 часов SASRec cache,
16 часов CPU cache; `TRAIN_TIME`, `CACHE_GPU_TIME`, `CACHE_CPU_TIME` имеют
приоритет над этими default. Из 20 завершённых sampled15 CE на H200: fit
3.01–11.12 ч при 1 295 282 targets. В GFlow sampled50 — 4 162 653 targets;
простое масштабирование даёт примерно 9.7–35.7 ч при измеренной скорости.
48 часов выбраны с запасом; это не гарантия времени A100, поскольку скорость
CE на нём ещё не измерена, а число эпох определяется early stopping.
Больший лимит можно задать явно. При выборе лимита учтены
[правила ресурсов](https://hpc.hse.ru/llms.txt) и
[настройки приоритета Slurm](https://slurm.schedmd.com/priority_multifactor.html#general).
Ресурсы и параллелизм задаются через переменные окружения; полный список:

```bash
bash scripts/slurm/submit_ce_distill_gpt2_yambda.sh --help
```

Например, `TRAIN_TIME=5-00:00:00` меняет лимит обучения,
`MAX_PARALLEL=1` ограничивает каждый массив одной задачей.

Кеши наград и таблица перевода IDs учителя для подвыборок по умолчанию:

```text
artifacts/exp7-ce-distillation-gpt2-yambda/<dataset>/reward_factors/<component>/
artifacts/exp7-ce-distillation-gpt2-yambda/<dataset>/source_assets/
```

Метрики, логи и обученные веса:

```text
experiments/exp7-ce-distillation-gpt2-yambda/<dataset>/<combo>/<mode>/pow_<p>/rho_<r>/seed_42/
```

Для произведения трёх наград `<combo>/` опущен.
Для sampled50 в run-путь после `<dataset>/` добавляется `sampled50_seed42/`,
а в cache-путь эта ветка добавляется перед `<dataset>/`, как показано выше.
`<mode>` — `sampled_n3` или `curriculum_smooth_is_n3_epochs5_qmix0p1`.
Внутри находятся `result.json`, `training_history.json`,
`training_log.jsonl`, `calibration.json`, конфигурации и
`checkpoints/{best,last}.pt`. Baseline сохраняется отдельно в
`<dataset>/baseline/init_metrics.json`.

Сводки по завершённым запускам:

```bash
python scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py --aggregate-only
python scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py \
  --aggregate-only --sampling curriculum_smooth_is
```

`metrics.csv` и `summary.json` появляются в
`experiments/exp7-ce-distillation-gpt2-yambda/ablations/<mode>/`.
При изменении сетки или параметров семплирования передайте те же флаги.
Для новой сетки sampled50:

```bash
for sampling in reward curriculum_smooth_is; do
  python scripts/exp7-ce-distillation/ce_distill_gpt2_yambda.py \
    --aggregate-only --datasets yambda_500m_listens --train-random-n 50 \
    --train-subset-seed 42 --sampling "$sampling"
done
```

Сводка sampled50 сохраняется в отдельной ветке, чтобы не перезаписать
сводку sampled15.
Общий `report_ce_results.py` также учитывает корень GPT-2 Yambda.
