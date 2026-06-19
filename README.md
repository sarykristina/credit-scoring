# Credit Scoring: Card Transaction History

Решение задачи прогнозирования дефолта клиента по истории кредитных/карточных записей.

## Структура

```text
.
├── data/                  # сюда положить исходные файлы соревнования
├── outputs/               # сюда сохраняются новые сабмиты
├── submissions/           # готовый проверенный submission.csv
├── work/                  # промежуточные признаки, не коммитятся
├── src/                   # код решения
├── requirements.txt
└── run_pipeline.sh
```

## Данные

Положите в папку `data/` файлы:

```text
train_data.parquet
test_data.parquet
train_target.csv
sample_submission.csv
```

Исходные parquet/csv с данными не включены в репозиторий, чтобы не раздувать GitHub.

## Быстрый результат

Готовый файл для отправки уже лежит здесь:

```text
submissions/submission.csv
```

## Полное воспроизведение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash run_pipeline.sh
```

Итоговый файл появится в:

```text
outputs/submission_lgbm.csv
```

## Модель

Финальная модель использует:

- агрегаты по клиенту;
- дополнительные распределения категориальных признаков;
- признаки последних записей истории;
- LightGBM с ROC-AUC validation control.

Проверенный локальный результат лучшей версии: `0.772153` ROC-AUC.
