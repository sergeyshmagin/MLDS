# UrbanStay: прогноз отмены бронирования

Модель предсказывает вероятность отмены брони и переводит прогноз в деньги через метрику
Incremental Revenue (IR). Проект выполнен на данных PostgreSQL сети отелей UrbanStay:
`hotel_bookings` (35 341 бронь) и `hotel_reviews` (25 177 отзывов).

## Состав репозитория

```
SPRINT_18/
├── bbab3c3f-71b2-439b-b484-121cdb98ce67.ipynb  тетрадь проекта
├── feature_pipeline.py                          очистка, объединение, признаки, бизнес-метрики
├── requirements.txt                             точные версии окружения
├── README.md
└── artifacts/
    ├── feature_pipeline.py                      копия модуля рядом с моделью
    ├── model.joblib                             Pipeline (препроцессор + откалиброванная модель) и порог
    └── model_meta.json                          гиперпараметры, порог, метрики, список колонок
```

## Установка

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

На JupyterHub Практикума первая ячейка тетради сама доустанавливает недостающие пакеты
(`psycopg2-binary`, `optuna`, `lightgbm`, `catboost`, `shap`, `numba`) и при необходимости
понижает SQLAlchemy до ветки 1.4: `pandas 1.2` несовместим с SQLAlchemy 2.x.

## Инференс на новых данных

```python
import joblib
import pandas as pd
import feature_pipeline as fp

artifact = joblib.load("artifacts/model.joblib")
pipeline, threshold = artifact["pipeline"], artifact["threshold"]

bookings = fp.clean_bookings(new_bookings_raw)   # сырая выгрузка hotel_bookings
reviews = fp.clean_reviews(reviews_raw)          # сырая выгрузка hotel_reviews
features = fp.build_features(fp.join_reviews(bookings, reviews))

probability = pipeline.predict_proba(features)[:, 1]
risky = probability >= threshold                 # брони, по которым стоит принимать меры
```

Порог хранится вместе с моделью: он подобран по максимуму IR и не равен 0.5.
Экономические параметры (`AvgRev`, `CostFP`, `LostRev`, `PerRebooking`) заданы константами
в `feature_pipeline.py`; при их пересмотре порог нужно пересчитать функцией
`fp.threshold_scan` на свежей калибровочной выборке.

## Ограничение данных

Отзыв в `hotel_reviews` есть у всех состоявшихся броней и ни у одной отменённой, поэтому
связывать таблицы по клиенту нельзя: любой признак, полученный через `booking_id` таблицы
отзывов, однозначно выдаёт целевую переменную. Объединение выполняется только по дате -
каждой брони достаётся последний отзыв, известный отелю на момент её оформления.
