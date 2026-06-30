# Sprint 14 - артефакты модели (BikeSouth)

Бандл лучшей модели прогноза почасового спроса на велосипеды. Лучшая модель -
**KNN** (`n_neighbors=11`, `weights=distance`, `p=1`) в sklearn Pipeline с
кастомным трансформером признаков.

## Результат на тесте

| Модель | RMSE | MAE | R2 | Отрицательные прогнозы |
|--------|------|-----|-----|------------------------|
| Линейная регрессия (baseline) | 411.45 | 312.53 | 0.586 | 147 |
| **KNN (финальная)** | **296.16** | **200.68** | **0.786** | **0** |

RMSE -28%, MAE -36%, R2 +0.20 относительно baseline.

## Состав

```
SPRINT_14/
├── requirements.txt            точные == версии Практикума (Python 3.9)
└── artifacts/
    ├── model.joblib            финальный KNN-пайплайн (сохранён под sklearn 1.6.1)
    ├── feature_pipeline.py     модуль признаков (нужен для joblib.load)
    └── model_meta.json         best_params, метрики, список колонок, random_state
```

## Окружение

Современный стек: **scikit-learn 1.6.1, numpy 1.26.4, pandas 2.3.3** (см.
`requirements.txt`). Под ним сериализован `model.joblib` и на нём же тетрадь
загружает предоставленный baseline.

## Инференс

```python
import joblib, pandas as pd
import feature_pipeline as fp          # должен быть рядом (есть в artifacts/)

model = joblib.load("artifacts/model.joblib")
raw = pd.read_csv("новые_данные.csv").rename(columns=fp.RENAME_MAP)
pred = model.predict(raw.drop(columns=[fp.TARGET], errors="ignore"))
```

Весь препроцессинг (импутация, масштабирование, OHE, инженерные признаки)
выполняется внутри пайплайна.
