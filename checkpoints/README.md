# Чекпоинты

```text
checkpoints/
  exp1/<model>/<dataset>/best.pt
  exp2/
    loo_split/
      initial/<model>/<dataset>/best.pt
      sft-finetuned/<model>/<dataset>/best.pt
    global_split/
      initial/<model>/<dataset>/best.pt
```

`exp1` содержит основные модели со стандартными параметрами.
В `exp2` данные разделены по протоколу: `loo_split` — leave-one-out,
`global_split` — глобальное временное разбиение. `initial` содержит исходные
затюненные модели, `sft-finetuned` — первое SFT-only дообучение этих моделей.
Повторное SFT из exp8 сюда не включается.

Для **GPT-2 Yambda** внутри папки датасета указан режим данных:

| Split / этап | Датасет | Папка режима | Основная модель |
| --- | --- | --- | --- |
| loo_split / initial | Yambda50M | `last_128` | trial36 |
| loo_split / initial | Yambda500M | `last_128` | trial4 |
| loo_split / initial | оба Yambda | `full_history` | legacy-базы |
| loo_split / sft-finetuned | Yambda50M | `last_128` | первый SFT от trial36, seed42 |
| loo_split / sft-finetuned | Yambda500M | `sampled_50` | первый SFT от trial4, seed42 |
| global_split / initial | Yambda50M | `full_history` | выбранная глобальная база |

Например: `exp2/loo_split/initial/gpt2/yambda_50m_listens/last_128/best.pt`.
Режим global GPT-2 Yambda50M определён по сохранённому `study_config.json`
(`train_target_policy=all`); он не является last-128. GPT-2 sampled-15
локально нет.

В каждой конечной папке хранится один основной `best.pt` и его метаданные.
Источник, фактический seed (если известен), режим и SHA-256 записаны в
`provenance.json`. Ссылки на конкретные seed читают выбранную основную модель.

У TIGER LOO Yambda50M выбран первый SFT `matched`, у Yambda500M —
`matched-r15`, оба seed42 и lr=3e-05. Исходная база TIGER Yambda500M сохранена
после дообучения до сходимости. Первый SFT TIGER Beauty2023 содержит
`best.pt` и `best.pt.json`, seed42. Для продолжения этого
Beauty2023 в exp8 дополнительно нужен полный исходный протокол/результаты
запуска: они не восстанавливаются предположениями из одних весов.

В `global_split` сохранены 13 подтверждённых основных моделей. Для GPT-2
Yambda500M и TIGER Beauty2023 основной global-split чекпоинт не выбран.
Глобальных SFT-only весов локально нет.

Скрипты читают входные основные веса только из этой структуры. Если модели
нет, они сообщают об отсутствии по текущему пути. Архив не используется как
запасной источник весов. Per-seed snapshots, `last.pt`, trials и журналы
обучения сохраняются в `experiments/`; публикация основной модели сохраняет
описанную структуру.

`configs/tiger_beauty_grpo.json` и
`configs/tiger_beauty_gflow_mix_curriculum_example.json` — шаблоны: поля из
`required_checkpoints` нужно заполнить путями к выбранным основным моделям.
`run_tiger.py` проверяет их до подготовки данных и создания запуска.
Архитектуру, настройки данных и semantic IDs в шаблоне нужно согласовать с
выбранной моделью; отсутствующие DPO/SASRec веса автоматически не заменяются.

Журналы запусков находятся в `experiments/`. Пути в сохранённых протоколах
служат идентификаторами запусков и не используются как места загрузки весов.

Словарь semantic-ID должен соответствовать обучению модели и храниться в
`datasets/semantic_ids/`. Бинарные файлы от 100 MiB остаются вне Git;
метаданные сохраняются в Git.
