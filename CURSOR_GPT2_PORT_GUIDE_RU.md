# Cursor guide: корректность данных и ускорение GPT2 в `rl_ablations`

Дата аудита: 21 октября 2026 года.

## 0. Задание и границы изменений

Рабочая ветка — `rl_ablations` репозитория `researcher/flowrec`. Требуется перенести отдельные исправления и оптимизации из `origin/main`, сохранив реализацию SDPO и MiniOneRec/GRPO, существующие checkpoint и протокол экспериментов.

Проверенные снимки:

```text
TARGET_RL = 08b7957056a31994069f3b49d9500bef2dc51952
DONOR_MAIN = 79b1833f76459fdc62315d8a2472b8ce3e25bff6
```

Состояние локальной рабочей копии на обучающем узле неизвестно. Перед реализацией сравни фактический checkout с этими SHA: локальные изменения могли уже исправить часть описанного.

**Не делай merge всей ветки и не заменяй целиком `scripts/run_tiger.py`.** В RL-ветке этот runner умеет запускать GPT2 и содержит маршрутизацию SDPO/GRPO; соответствующий builder в donor-main устроен иначе. Переноси функции и небольшие согласованные изменения с тестами.

Раздели работу на независимые коммиты:

1. Исправление обработки конфигурации и построения обучающих примеров.
2. Дифференцируемый KV-кеш в модели, ещё без изменения методов.
3. Подключение оптимизированного scoring к SDPO и GRPO за feature flag.
4. Ограничение памяти, калибровка и управление microbatch.
5. Тесты интеграции и измерение производительности.

Первый пункт меняет фактические training contexts и требует новой серии экспериментов. Пункты 2–4 должны сравниваться с неускоренной реализацией **на одинаковых, уже исправленных данных**.

## 1. Что подтверждено аудитом

### 1.1. `train_prefix_from_dense` в RL-конфигах сейчас не исполняется

В проверенных конфигах:

```text
configs/MiniOneRec/GPT2/gpt2_beauty_grpo.json
configs/DPO/GPT2/gpt2_beauty_sdpo.json
```

стоит:

```json
{
  "train_example_layout": "prefix_target",
  "train_prefix_from_dense": true,
  "max_history_len": 20,
  "train_target_policy": "all"
}
```

Но в TARGET_RL:

- `scripts/run_tiger.py::build_data_config` не передаёт `train_prefix_from_dense` ни в Amazon-, ни в Yambda-конфигурацию;
- `AmazonReviewDataConfig` не имеет такого поля;
- `SequenceSplitConfig` не реализует этот режим;
- `flowrec/data/dense_prefix.py` отсутствует;
- ветка `prefix_target` в `build_leave_one_out_examples` строит обычные ограниченные скользящие префиксы.

Следовательно, конфиг обещает один протокол, а опубликованный runner запускает другой. Это подтверждённая ошибка исполнения конфигурации, а не только отличие гиперпараметров.

Сам по себе rolling-prefix objective допустим. Ошибка состоит в незаметной подмене заявленных контекстов и отсутствии сопоставимости с dense-trained checkpoint. Из этого не следует, что прежние метрики сфальсифицированы или что доказана причина их падения.

### 1.2. Одинаковые targets и число примеров не доказывают одинаковые данные

Пусть последовательность пользователя — `A B C D E F G H I`, где `H` — validation target, `I` — test target. Для train остаётся `A..G`. При длине окна и stride 3 dense-builder, якорящий окна от конца train-последовательности, создаёт:

```text
inputs [A B C], labels [B C D]
inputs [D E F], labels [E F G]
```

Развёртка исходных окон:

```text
[A]       -> B
[A B]     -> C
[A B C]   -> D
[D]       -> E
[D E]     -> F
[D E F]   -> G
```

Обычные скользящие префиксы:

```text
[A]       -> B
[A B]     -> C
[A B C]   -> D
[B C D]   -> E
[C D E]   -> F
[D E F]   -> G
```

Targets и число строк совпадают. Контексты для E/F различаются, как и positional embeddings. Для настоящих окон 20/50 эффект зависит от длины пользовательских историй; его долю надо измерить на данных, а не угадывать. Все показанные токены находятся в прошлом: это не пример утечки validation/test.

### 1.3. Базовый causal shift модели не требуется переписывать

В обеих ветках GPT2:

- компактный left-padded prompt строится только из history;
- `position_ids = cumsum(attention_mask)-1`, а позиции padding зануляются;
- target token 0 предсказывается логитом последнего реального history token;
- prefix-target logits выбираются с позиции `prefix_width - 1`;
- dense collator уже делает `inputs = stream[:-1]`, `labels = stream[1:]`;
- первые `SID_len - 1` меток, восстанавливающие первый context item, выключены из loss;
- SFT использует уже сдвинутые labels и их маски. Ещё один сдвиг делать нельзя.

По статическому сравнению этих участков не найдено ошибки off-by-one. Требуются динамические тесты на реальном checkout.

### 1.4. Не все checkpoint в main были обучены на dense-окнах

В main существуют разные протоколы:

- legacy exp2 GPT2 SFT: dense-causal windows;
- exp4/exp7 fine-tuning соответствующих checkpoint: dense-derived prefixes;
- новые Yambda `--match-tiger` studies и matched trial-36: rolling prefixes, last-128 targets и отдельный preprocessing audit.

**Источником истины должна быть конфигурация конкретного initial checkpoint.** Нельзя включать dense-derived контексты для всех GPT2 только по названию архитектуры. Для matched-prefix checkpoint требуется сохранить его rolling-prefix протокол.

### 1.5. Дополнительная проверка Yambda subset

В RL-конфиге `gpt2_yambda_50m_listens_grpo.json` есть `finetune_subset` с 15 случайными targets из последних 128. В просмотренном `run_tiger.main` после `data.setup()` сразу создаётся `train_dataloader`; вызова применения этого блока нет. Проверь весь локальный launch path. Если блок не применяется, либо реализуй его явно, либо заверши запуск ошибкой при наличии неподдерживаемого параметра.

Это отдельный вопрос от выбора dense/rolling контекста. Не переносить автоматически число 15 в новый matched Yambda50-протокол: у него может быть другой предусмотренный набор targets.

## 2. Этап 0: зафиксировать исходный запуск

В отдельной локальной рабочей копии, не затрагивая незакоммиченные изменения:

```bash
git status --short
git fetch origin main rl_ablations
# Не использовать git reset --hard или git clean.
git rev-parse HEAD origin/main origin/rl_ablations
git diff 08b7957056a31994069f3b49d9500bef2dc51952 \
         79b1833f76459fdc62315d8a2472b8ce3e25bff6 -- \
  flowrec/models/gpt2.py flowrec/data scripts/run_tiger.py \
  flowrec/ft_methods/calibration.py
```

Для сравнения содержимого двух снимков здесь нужен прямой diff двух SHA, а не только diff main относительно merge-base.

В manifest запуска запиши:

- commit, dirty diff и фактические `__file__` импортированных model/data/method modules;
- Python/PyTorch/Transformers/CUDA версии, GPU и attention backend;
- SHA256 initial/student/reference/teacher checkpoint и semantic-ID assets;
- dataset config после обработки, context policy, train-target subset и его seed;
- ordered hashes item2id, semantic catalog и complete validation/test examples;
- сериализованный фактический generation config, включая `renormalize_logits`;
- microbatch, accumulation, effective target batch и единицы curriculum.

Нельзя считать checkpoint совместимым только по размерностям матриц. Проверяй соответствие raw item IDs, internal IDs, SID и порядка каталогов student/teacher/data.

## 3. Этап 1: исправить data pipeline

### 3.1. Файлы-доноры из main

```text
flowrec/data/dense_prefix.py
flowrec/data/sequence_builder.py
flowrec/data/datamodule.py
flowrec/data/yambda.py
scripts/run_tiger.py             # только относящиеся к данным фрагменты
```

Ориентиры: `dense_reference_config`, `prefixes_from_dense_windows`, `expand_dense_training_split`, `dense_prefix_signature`, `example_digest`, `validate_dense_prefix_match_config`.

### 3.2. Протянуть флаг до реально построенных splits

Добавить согласованную поддержку `train_prefix_from_dense: bool = False` в dataconfig обоих источников и необходимые validators/поля split config. В `run_tiger.build_data_config` явно передавать флаг. Сохранять его в `resolved_config` и summary вместе с интерпретацией контекста.

При `True` порядок должен быть следующим:

```text
preprocessing и factorization
    -> reference dense splits с исходными window_len / stride
    -> catalog filtering/remap на полном reference train
    -> semantic-ID/catalog alignment
    -> train popularity statistics на исходном train stream
    -> развёртка активных dense labels в отдельные prefix-target examples
    -> target selection / deterministic subset без перемещения окон
    -> dataset и dataloader
```

Перенести соответствующие проверки из main, а не писать вторую несовместимую реализацию окон.

Validation/test НЕ разворачивать и НЕ пересобирать из усечённого fine-tuning train. Их ordered полные примеры, каталог, timestamps и collated tensors должны совпадать с reference pipeline. Сохранять auxiliary interaction store: сокращение metadata не должно удалять события, от которых зависят rewards.

### 3.3. Последние N targets и overlap

Сначала определить исходные окна и активные labels. Затем выбрать последние N targets или фиксированное подмножество. История каждого оставшегося target начинается на исходной границе окна. Не отрезать сначала train-последовательность до N и не строить окна заново.

Для `stride < window_len` сохранить label masks, чтобы один target не превращался в несколько обучающих примеров. Для global-time использовать его исходный cutoff, не правила leave-one-out.

При subset сохранять соответствие `example_indices` фактическим строкам reward/reference caches. Размер и порядок caches должны соответствовать выбранному subset. Teacher histories должны точно совпадать с student histories.

### 3.4. Кеши и старые результаты

Добавить context signature и subset signature в processed-data/reward/reference-cache fingerprints; обновить cache schema. Ordered hash должен учитывать history, target, timestamp и SID, а не только user/target IDs. Проверить сохранение/восстановление `dense_prefix_audit` при reload.

Старый cache нельзя принимать только потому, что число строк совпало. Старые завершённые rolling-prefix запуски не должны автоматически считаться завершёнными dense-prefix запусками. Создать отдельный output root, например `gpt2_dense_prefix_v2`, без перезаписи прежних результатов.

### 3.5. Аудит до обучения

Добавить data-only режим runner (или отдельный audit script), который:

1. Строит legacy, reference-dense и исправленный prefix pipeline на одинаковом preprocessing.
2. Проверяет равенство множества/кратности активных targets, а также context для каждого target внутри соответствующего окна.
3. Проверяет complete validation/test hashes, item2id, semantic IDs и auxiliary evidence.
4. Записывает долю train targets с изменившейся относительно legacy историей, распределение context length и число supervised tokens.
5. Проверяет reload processed cache и несовместимость старого cache.

131413 targets для Beauty и 116989 для Instruments упомянуты в donor-аудитах. Это ориентир для того же preprocessing, НЕ достаточное доказательство корректности: число targets может совпасть при разных histories.

## 4. Этап 2: дифференцируемое переиспользование истории

### 4.1. Что действительно добавлено в main

В `flowrec/models/gpt2.py` перенести:

```text
cache_history_encoding_grad
_candidate_logits_from_grad_history
ветку _gpt2_history_grad_state в candidate_token_logits
```

Donor-main использует эти функции в `flowrec/ft_methods/reward_distillation.py`. Это инфраструктура CE-distillation; она не означает, что SDPO/GRPO ускорятся без изменения своих call sites.

Схема:

```text
policy history -> один differentiable prefill -> immutable K/V tensors
                                            -> первые logits для target token 0
N кандидатов -> свежий cache на каждый decoder call -> только L-1 новых токенов
```

У исходного пути history повторяется для каждого кандидата. Новый путь вычисляет history один раз для модели на microbatch, сохраняя градиент через history. Это убирает избыточные вычисления, но размер выигрыша в throughput/VRAM нужно измерить.

### 4.2. Контракт кеша

- В policy-prefill нет `no_grad`, `inference_mode` и `detach`.
- `eval()` допустим и не запрещает autograd; режим dropout обсуждается отдельно ниже.
- Хранить immutable ссылки на K/V tensors, prefix mask и history-last logits.
- Каждый decoder call создаёт свежий `DynamicCache` из этих tensors. Не дописывать candidate tokens в общий cache; не делать `deepcopy` non-leaf graph tensors.
- Маска decoder attention покрывает physical prefix width + L-1 новых токенов. Positions реальных токенов вычисляются по маске, а не по общему padded width.
- Для L=1 ответ — сохранённые first-token logits, без дополнительного decoding.
- Candidate masks и masked -100 должны поддерживаться без выхода за vocabulary.
- Policy и reference используют РАЗНЫЕ prefill: нельзя переносить cache между моделями.
- Cache живёт не дольше одного microbatch/backward; после изменения параметров он недействителен.
- Остальные GPT2 interfaces, state_dict keys, vocab и generation semantics не менять.

В RL уже существует `cache_history_encoding` под `@torch.no_grad()`. Это inference helper; его нельзя подставить в trainable policy path. Кроме того, существующая inference-ветка в `candidate_token_logits` используется только при `candidate_token_mask is None`. Простое добавление кеша в batch не гарантирует, что call site его использует.

### 4.3. Dropout: обязательная граница эквивалентности

Когда dropout активен, отдельные history copies получают разные маски. Общий prefill делит одну маску между кандидатами. Для nonlinear SDPO loss это может изменить даже ожидаемый objective, а не только скорость или variance.

Начальный безопасный режим:

- GRPO policy scoring: оставить существующий eval-режим; cache сохраняет градиент.
- Frozen reference: eval + no_grad, кешировать безопасно.
- SDPO policy: при активном dropout использовать старый путь. Кеш включать только для отдельно согласованного deterministic preference scoring / dropout=0.
- SFT: собственный прежний forward на исходном batch; не переиспользовать preference cache с другим train/eval-режимом.

Отключение dropout — отдельное изменение алгоритма/регуляризации. Нельзя включить его незаметно под названием «ускорение». Если вводится eval-scoring для SDPO, сравнить accelerated/unaccelerated оба в этом режиме, а затем отдельно сравнить его со старым dropout-протоколом.

## 5. Этап 3: подключить кеш к SDPO

### 5.1. Почему одного переноса GPT2-функций недостаточно

В TARGET_RL `SDPOMethod.compute_loss` вызывает:

```text
model.sequence_logprobs(chosen)
model.score_items(rejected)
```

Generic `FlowRecModel.score_items` разворачивает историю B*K и вызывает `sequence_logprobs`. У GPT2 тот вызывает `_forward_prefix_target`, обходя cache-aware `candidate_token_logits`.

Нужно изменить именно GPT2 scoring path SDPO либо добавить корректный специализированный override модели. Generic fallback для остальных моделей оставить без изменения.

### 5.2. Семантика scoring

Собрать выбранные candidates ОДИН раз:

```python
all_ids = torch.cat((chosen_ids[:, None, :], rejected_ids), dim=1)
all_mask = torch.cat((chosen_mask[:, None, :], rejected_mask), dim=1)
```

Для допустимого deterministic policy path:

```python
policy_context = model.cache_history_encoding_grad(batch)
policy_token_lp = model.score_item_tokens(
    policy_context,
    candidate_token_ids=all_ids,
    candidate_token_mask=all_mask,
)
policy_lp = policy_token_lp.float().sum(dim=-1)
chosen_logp, rejected_logp = policy_lp[:, 0], policy_lp[:, 1:]
```

Это схема, а не готовая замена всего метода: нужны mode guard, feature flag и generic fallback. Для reference создать отдельный context внутри `torch.no_grad()`. Можно использовать тот же tensor-based prefill helper под `no_grad()` либо расширить inference-helper для masks; запрещено передавать policy cache в reference.

**Сохранить semantic-vocabulary normalization.** Исходный SDPO использует `sequence_logprobs`, а не local valid-actions normalization. Поэтому вызывать `score_item_tokens`, НЕ `score_item_tokens_valid_actions`. Иначе меняется математический objective.

По SID-токенам нужна сумма, как в исходном `sequence_logprobs`, не среднее. Не менять beta, sampler, frozen reference, количество negatives и softmax preference formula.

При chunking сохранить все B*(K+1) scalar sequence scores, затем выполнить один общий `logsumexp` по chosen и всем negatives. Сумма/среднее отдельных chunk-wise SDPO losses не эквивалентна исходной функции.

## 6. Этап 3b: подключить кеш к GRPO

В `MiniOneRecMethod` оставить неизменными generation, reward, advantages и old-policy semantics. После получения фиксированных trajectories:

1. В существующем eval-scoring контексте сделать один policy-prefill с autograd.
2. Посчитать те же per-token probabilities через `score_item_tokens_valid_actions`, когда это исходный выбранный scorer.
3. Сохранить old probabilities detached; не давать градиент знаменателю ratio.
4. Reference-prefill и scoring выполнять отдельно, eval/no_grad.
5. Не менять clipping, token averaging, group averaging, reward normalization и SFT term.

Не смешивать этот перенос с изменениями beam/sample, top-k/top-p, GT injection, decoder renormalization или rollout reuse. Согласованность sampling/scoring — отдельная correctness-проверка. Изменение decoder требует переоценки исходного checkpoint, а не объявления нового результата выигрышем обучения.

Убирать повторный old-policy scoring через `policy_lp.detach()` можно только как позднюю оптимизацию с доказанным одношаговым инвариантом old=current, одинаковой precision/mode и отсутствием обновлений между вычислениями. Для первого переноса этого не требуется.

## 7. Этап 4: управление памятью и калибровкой

### 7.1. Candidate chunking и activation checkpointing

Из `reward_distillation.py::_exact_prefix_ce` переносится техника небольших decoder chunks и `checkpoint(..., use_reentrant=False)`, а НЕ CE objective или агрегация teacher q по полному каталогу.

Chunking без checkpoint уменьшает размер временного forward, но может оставить графы всех chunks живыми до backward. Не обещать уменьшения training peak memory только по снижению chunk size: измерить saved activations и пик памяти.

При использовании checkpoint:

- передавать nested tensor state, не mutable shared HF cache;
- сохранять RNG и тот же autocast/mode при recomputation;
- **особенно важно:** GRPO временно переводит policy в eval и возвращает train до backward. Checkpointed функция при recomputation должна снова работать в том же eval-режиме. Иначе dropout включится только в backward recomputation;
- использовать явный context manager внутри функции или non-reentrant `context_fn`, восстанавливая исходные training flags модулей;
- сравнить logits, loss и все gradients с вариантом без checkpoint;
- chunking не должен менять mean denominator. Неполный последний chunk имеет свой фактический размер.

Checkpointing обменивает дополнительные вычисления на память; сам по себе это не обязательное ускорение. Оценивать, позволяет ли он увеличить полезный microbatch и сократить общее время optimizer update.

### 7.2. Последовательная FP32-калибровка

Donor: `flowrec/ft_methods/calibration.py::calibrate_sequential_variants` и общий helper статистик градиентов.

Идея — держать не два больших forward graphs одновременно, а вычислить gradient reference-компоненты, освободить граф, затем вычислить вторую компоненту. Сохраняются detached gradient vectors для norm/cosine.

Переносить с условиями:

- обе компоненты используют те же данные/candidates и согласованные RNG/modes;
- если исходный код получает оба loss из одного forward, нельзя просто выполнить два независимых random-sampling forward и назвать их эквивалентными;
- calibration FP32 и число батчей сохраняются;
- порядок reference/component проверяется отдельно: RL калибрует SFT относительно SDPO/GRPO, а CE-скрипты могут калибровать противоположный коэффициент;
- после калибровки восстановить RNG, модельные режимы, отсутствие `.grad` побочных эффектов;
- сравнить новый и старый weight на одном fixed candidate set;
- перенос на меньший microbatch меняет статистику norm. Для исследовательского сравнения фиксировать и calibration microbatch либо калибровать accumulated effective batch в отдельном протоколе.

### 7.3. Batch size, OOM и curriculum

Donors: `scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py`, `gflow_finetune_gpt2_yambda.py`; `batch_settings_for_k` в общем `gflow_finetune_yambda50m.py`.

Донор ограничивает candidate-expanded batch и сохраняет effective batch при уменьшении microbatch. Например, 32*64, 64*32 и 128*16 дают 2048 prefix targets на update, но реальные изменения времени/памяти надо измерить. Для dense SFT единица batch — окно, а не target, поэтому произведение batch*accum само по себе уже не означает 2048 targets.

На OOM нельзя продолжить с частично накопленными градиентами или частично изменённым optimizer state. Донор может перезапускать run с исходного checkpoint; для RL выбери явную проверенную политику: целиком перезапустить попытку либо восстановить последний согласованный checkpoint+RNG. Фиксировать факт retry, реальные microbatch/accum и число успешных updates.

При изменении microbatch сохранить schedule curriculum по числу обработанных training examples / исходному времени обучения. Старый `curriculum_steps` в SDPO считается в microbatches: оставить то же число при увеличении batch означает изменить сам curriculum. На неполном accumulation tail правильно учитывать фактический denominator; тестировать equivalent effective-batch gradient.

Не добавлять «обработку OOM», молча пропускающую трудные батчи: это меняет обучающее распределение.

### 7.4. CPU/RAM pipeline

В main есть `metadata_ids_only`, сокращающий загрузку metadata и создание больших Python dicts. Его можно перенести для запусков, которым не нужны тексты, сохранив то же множество raw IDs и фильтрацию каталога.

Fingerprint processed-cache должен учитывать context/subset/protocol. Не отключать фильтры, auxiliary events, metadata intersection или популярности ради скорости.

Tensorized prefix batching, prefix trie, compact padding и inference KV-cache уже присутствуют в RL-коде. Сначала проверить, что они реально используются; повторная реализация не является новым переносом. Fast greedy generation для K=1 не ускоряет автоматически validation beam20.

## 8. Feature flags и совместимость

Предлагаемые НОВЫЕ настройки (не существующие сейчас API):

```json
{
  "method": {
    "gpt2_shared_history_cache": false,
    "gpt2_candidate_chunk_size": null,
    "gpt2_checkpoint_candidates": false
  }
}
```

Выбранные имена можно изменить, но необходимо добавить их в dataclass, from_dict, runner validation и resolved config. Оптимизации выключены по умолчанию до прохождения тестов. Не повторять проблему silently ignored `train_prefix_from_dense`.

При активном dropout SDPO policy-cache должен явно отказываться от ускоренного пути или требовать отдельного согласованного режима; fallback должен логироваться. Для non-GPT2 сохранить старый путь. Никакие веса архитектуры/размерности или state_dict имена не должны измениться.

## 9. Тесты приёмки

### 9.1. Donor regression tests

Изучить и адаптировать, не импортируя весь несвязанный CE/GFlow стек без необходимости:

```text
tests/test_gpt2_amazon_dense_prefix.py
tests/test_gpt2_yambda_dense_prefix.py
tests/test_sft_control_gpt2_amazon_dense.py
tests/test_gpt2_reward_distillation.py
```

### 9.2. Подготовленные contract tests

В архиве есть `tests/test_gpt2_rl_port_contracts.py`. После помещения в root проекта:

```bash
python -m pytest tests/test_gpt2_rl_port_contracts.py -q
```

Эти тесты проверяют parser flag, окно/targets, masked causal alignment, cached scoring/градиенты и объединённый SDPO-scoring. На старой RL-ветке часть обязана падать: это red tests для переноса.

**Файл подготовлен и проверен синтаксически, но в среде аудита не исполнен:** здесь отсутствует Transformers и рабочий checkout с зависимостями. Он не заменяет integration tests самих SDPOMethod/MiniOneRecMethod и настоящий data audit.

### 9.3. Дополнить интеграционными тестами

Обязательны:

1. Полные Amazon/Yambda data pipeline, cold filter on/off, dense window overlap, short histories, last-N и deterministic subset, сохранение held-out/catalog/auxiliary store, processed-cache reload.
2. Dense SFT vs expanded-prefix SFT: одинаковая сумма CE на активных tokens и все parameter gradients при dropout=0; не среднее средних окон с разным числом targets.
3. GPT2 cached/uncached: неодинаковые длины histories, left padding, L=1, masks, обе normalization conventions, повторное чтение кеша и отсутствие мутаций.
4. Настоящий SDPO: fixed negatives/reference; chosen/rejected log-probs, preference loss, SFT loss, total loss и все gradients с flag off/on; глобальный softmax при chunk sizes 1, 2, K+1.
5. Настоящий GRPO: fixed trajectories/rewards/old probabilities; per-token ratios/KL/advantages/loss/gradients off/on; reference без градиентов; history-параметры с ненулевыми gradients.
6. Отдельный тест режима checkpoint recomputation после восстановления `model.train()`: eval-forward должен оставаться eval при recomputation.
7. Finite gradients под целевой AMP precision; отсутствие пропущенных AMP updates скрытно за счётчиком.
8. Один одинаковый optimizer update off/on и checksum/close параметров; accumulation tail и OOM retry не теряют/не дублируют examples.
9. Модель после загрузки старого checkpoint выдаёт те же initial validation predictions при неизменном decoder.

Начальные FP32 tolerances для tiny-model parity: atol=2e-6, rtol=3e-4, как в donor-тестах; отдельно измерить численную погрешность на реальных моделях. Не ослаблять tolerance, чтобы скрыть detach/masking ошибки. Для dropout>0 bitwise equality не является обещанием кеша; нужна отдельная спецификация stochastic semantics.

## 10. Benchmark и rollout изменений

Сначала измерить CPU data build/calibration, затем GPU отдельно по этапам:

| Фаза | Что измерять |
|---|---|
| Data-only | время preprocessing, RAM, cache reload; полная идентичность требуемых данных |
| SDPO scoring | policy, reference, SFT, backward; K=3 и 8, короткие/длинные histories |
| GRPO scoring | generation отдельно от old/current/reference scoring; G=8 и 16 |
| Memory | peak allocated/reserved, saved activations, microbatch и accumulation |
| Training | секунды на optimizer update и targets/sec при одинаковых примерах |
| Evaluation | одинаковые beams, K, split, cutoff и prediction metrics |

Использовать warmup, CUDA synchronize для стендового измерения, несколько повторов и median/p90. Не синхронизировать GPU в каждом обычном train step ради логирования. Указать реальные версии библиотек, устройства и precision.

Три независимые проверки, не смешивать:

```text
legacy contexts + legacy scoring       -> исторический результат
corrected contexts + legacy scoring    -> эффект исправления данных
corrected contexts + accelerated scoring -> эффект оптимизации вычислений
```

После synthetic/integration tests выполнить Beauty SFT/SDPO/GRPO smoke с небольшим числом batches и отдельной выходной директорией. Затем один полный контрольный запуск. Только после него — перенос на остальные datasets/seeds.

Не обещать коэффициент ускорения без измерений. Общий выигрыш может ограничиваться generation, teacher scoring, Python dataloader или validation, даже если candidate forward заметно быстрее.

## 11. Что Cursor должен представить по завершении

- Список небольших коммитов и изменённых файлов, без несвязанных правок.
- Таблицу каждого config key: где читается, где применяется, где записывается в resolved config.
- Data audit: target/context counts/hashes, каталог, held-out и auxiliary evidence, до/после.
- pytest report с перечислением реально выполненных тестов и версий библиотек.
- Loss/gradient/optimizer-update parity report; отдельно stochastic/dropout оговорки.
- Benchmark off/on на одинаковом corrected data pipeline, без изменения evaluation/объектива.
- Перечень несовместимых старых caches и новый output root; старые файлы не удалять.
- Что ещё не проверено. Не объявлять работу законченной только по успешному импорту модели.

## 12. Проверенные источники

Все пути ниже относятся к двум SHA из начала документа. Для получения точного исходника:

```bash
git show 79b1833f76459fdc62315d8a2472b8ce3e25bff6:flowrec/data/dense_prefix.py
git show 79b1833f76459fdc62315d8a2472b8ce3e25bff6:flowrec/models/gpt2.py
git show 08b7957056a31994069f3b49d9500bef2dc51952:scripts/run_tiger.py
```

| Путь | Что проверено |
|---|---|
| `configs/MiniOneRec/GPT2/gpt2_beauty_grpo.json`, RL | Dense-prefix флаг в config, prefix layout, batch32×64 |
| `configs/DPO/GPT2/gpt2_beauty_sdpo.json`, RL | Тот же флаг, SDPO parameters и explicit dropout |
| `scripts/run_tiger.py`, RL, примерно 100–160 и 278–460 | Setup/loaders и игнорируемый data flag |
| `flowrec/data/datamodule.py`, RL, 1–200 | Поля dataconfig, отсутствие dense-prefix поддержки |
| `flowrec/data/sequence_builder.py`, RL, 390–615 | Sliding prefix и end-anchored dense windows |
| `flowrec/data/collators.py`, RL, 540–715 | One-token shift, token-label masks |
| `flowrec/models/gpt2.py`, обе ветки, примерно 280–570 | Совпадающий prefix/dense causal scoring |
| `flowrec/data/dense_prefix.py`, main | Развёртка активных labels, audit/signature |
| `flowrec/data/datamodule.py`, main, 430–545 | Catalog до target expansion, held-out сохранение |
| `scripts/exp4-reward-ablations/GPT2_DENSE_PREFIX.md`, main | Dense-derived протокол и data audits |
| `scripts/exp2-tuned-params/tuning_gpt2.py`, main | Legacy dense HPO и новый matched-prefix Yambda HPO |
| `flowrec/models/gpt2.py`, main, примерно 602–770 | Differentiable KV cache |
| `flowrec/ft_methods/reward_distillation.py`, main | Использующий cache метод; checkpointed chunks |
| `flowrec/ft_methods/calibration.py`, main | Sequential FP32 calibration |
| `flowrec/ft_methods/preference_sdpo/method.py`, RL, 65–210 | Необходимые scoring call sites |
| `flowrec/models/base.py`, RL, 75–170 | Sum log-probs и generic expanded-batch scoring |
| `tests/test_gpt2_reward_distillation.py`, main | Cache/gradient parity, masks, CPU autocast tests |

Первичные справочные страницы для checkpoint/cache (проверять поведение именно установленной версии):

```text
https://docs.pytorch.org/docs/stable/checkpoint.html
https://huggingface.co/docs/transformers/en/cache_explanation
```

Обычный inference-cache нельзя автоматически считать пригодным для обучения; требуется именно описанный graph-preserving путь и regression tests. В PyTorch non-reentrant checkpoint поддерживает nested tensor structures и autograd.grad; повторное вычисление должно сохранять forward semantics.
