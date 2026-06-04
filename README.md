# MLDS Sprint 13 — Возрастная классификация пользователей

Артефакты модели для проекта Sprint 13: классификация пользователей сети «Йети» на 5 возрастных категорий.

## Содержимое ветки

| Файл | Что это |
|---|---|
| `43cb4ff3-e668-4a18-aa22-aa1bc6312eb1.ipynb` | Основная тетрадь проекта со всеми этапами (EDA → подбор гиперпараметров → артефакты) |
| `artifacts/feature_pipeline.py` | Модуль с функцией сборки признаков `build_features` и `ManualStandardScaler`. Импортируется при загрузке модели. |
| `artifacts/model.joblib` | Сериализованный обученный пайплайн `ColumnTransformer + SVC(kernel="rbf", probability=True)`. |
| `artifacts/model_meta.json` | Метаданные: лучшие гиперпараметры, F1 на CV и тесте, список колонок, `RANDOM_STATE`. |
| `requirements.txt` | Версии библиотек, **зафиксированные под среду JupyterHub Практикума** (Python 3.9 + sklearn 0.24). |

## Метрики (SVC rbf, `C=1.0`, `gamma=0.05`, `probability=True`)

- F1_macro CV: **0.891 ± 0.011**
- F1_macro test: **0.892**
- Δ(test − cv): +0.002 (без переобучения)
- precision_test: 0.888, recall_test: 0.898
- Доля «<18 → 18+»: 8.9%

## Версии окружения

| Пакет | Версия |
|---|---|
| Python | 3.9 |
| pandas | 1.2.5 |
| numpy | 1.19.5 |
| scikit-learn | 0.24.2 |
| matplotlib | 3.4.3 |
| seaborn | 0.11.2 |
| statsmodels | 0.12.2 |
| joblib | 1.0.1 |

## Как запустить инференс

```python
import sys, joblib
sys.path.insert(0, "artifacts")          # чтобы найти feature_pipeline
from feature_pipeline import build_features

model = joblib.load("artifacts/model.joblib")
X = build_features(visits, ads, surf, cloud, primary)   # 5 сырых датафреймов
y = model.predict(X)
probs = model.predict_proba(X)                          # доступно благодаря probability=True
```

Для защиты несовершеннолетних: выставляйте порог по `probs[:, 0]` — показывайте 18+ креативы только если `P(<18) < 0.3`.
