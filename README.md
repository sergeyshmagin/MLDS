# Sprint 16. AutoValue AI: оценка стоимости автомобиля

Турнир трёх библиотек градиентного бустинга (XGBoost, CatBoost, LightGBM) на задаче
предсказания цены автомобиля с пробегом. Решение выбирается по комбинации базовых метрик
(MAE, RMSE, R²) и бизнес-метрик риска.

## Состав

```
SPRINT_16/
├── 51c3ffbf-518e-4434-b5db-bf800092ea64.ipynb   тетрадь проекта
└── requirements.txt                              версии окружения платформы
```

Исходные датасеты (`ds_s16_train_data.csv`, `ds_s16_test_data.csv`) в репозиторий не
выкладываются: они относятся к учебной платформе и лежат в `/datasets/`.

## Бизнес-метрики

При `error_ratio = (прогноз - факт) / факт`:

* **Overpricing Rate** - доля сделок с `error_ratio > 0.20`, риск выкупить машину дороже рынка;
* **Underpricing Loss** - сумма `факт - прогноз` там, где `error_ratio < -0.20`, упущенная выручка.

## Итоговая модель

`CatBoostRegressor` с категориальными признаками в нативном формате (отдельный препроцессор
не нужен) и параметрами, найденными Optuna:

```python
CatBoostRegressor(
    loss_function='MAE',
    learning_rate=0.03819,
    iterations=1587,
    depth=3,
    l2_leaf_reg=9.79647,
    random_seed=42,
    cat_features=['brand', 'fuel_type', 'transmission', 'color',
                  'service_history', 'insurance_valid', 'region'],
)
```

Результат на отложенном тесте (2 000 сделок): **MAE 75 696 руб., RMSE 94 817, R² 0.8748,
Overpricing Rate 13.45%, Underpricing Loss 19.39 млн руб.** Расхождение с валидацией не
превышает 4.5%, переобучения нет.

Дополнительно в тетради посчитан осторожный режим - тот же CatBoost с `Quantile:alpha=0.4`:
Overpricing Rate падает до 10.30% ценой роста упущенной выгоды на 74.8%.

## Запуск тетради

```bash
pip install -r requirements.txt
jupyter notebook 51c3ffbf-518e-4434-b5db-bf800092ea64.ipynb
```

Первая ячейка тетради сама доустанавливает бустинги, `shap` и `optuna`, если их нет, и
проверяет у xgboost версию, а не факт установки. Пути к данным определяются автоматически:
`/datasets/` на платформе, иначе `artifacts/` рядом с тетрадью.
