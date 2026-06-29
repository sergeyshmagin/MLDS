"""Feature engineering for the BikeSouth hourly-demand project (Sprint 14).

This module is the single source of truth for the custom transformers and the
column groups used by the notebook. The notebook imports from here so that the
custom classes keep a canonical ``__module__`` ("feature_pipeline") and the saved
joblib pipeline can be loaded back without monkey-patching ``__class__``.

Targets the JupyterHub Praktikum stack: Python 3.9, scikit-learn 0.24.2,
numpy 1.19.5, pandas 1.2.5 (hence ``OneHotEncoder(sparse=False)`` and manual
feature-name building instead of ``get_feature_names_out``, which appeared in 1.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Maps the raw CSV headers to snake_case names (the same names the original
# baseline pipeline used), so one rename serves both the baseline and our models.
RENAME_MAP = {
    "Temperature": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed_ms",
    "Visibility (10m)": "visibility_10m",
    "Dew point temperature": "dew_point_temperature",
    "Solar Radiation (MJ/m2)": "solar_radiation_mjm2",
    "Rainfall(mm)": "rainfallmm",
    "Snowfall (cm)": "snowfall_cm",
    "Seasons": "seasons",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
    "Time_Period_Evening": "time_period_evening",
    "Time_Period_Late Evening": "time_period_late_evening",
    "Time_Period_Morning": "time_period_morning",
    "Time_Period_Night": "time_period_night",
    "Rented Bike Count": "rented_bike_count",
}

TARGET = "rented_bike_count"

# The four one-hot columns that encode the time of day. "Daytime" is the implicit
# baseline period (all four are False) - information present but not stated.
TIME_BOOL_COLS = [
    "time_period_evening",
    "time_period_late_evening",
    "time_period_morning",
    "time_period_night",
]

# Column groups AFTER WeatherFeatureEngineer has run.
NUMERIC_COLS = [
    "temperature",
    "humidity",
    "wind_speed_ms",
    "visibility_10m",
    "dew_point_temperature",
    "solar_radiation_mjm2",
    "rainfallmm",
    "snowfall_cm",
    "temp_solar",          # temperature x solar radiation (the "+28 in the sun" case)
    "temp_dewpoint_gap",   # temperature - dew point (air dryness / mugginess)
]
CATEGORICAL_COLS = ["seasons", "holiday", "functioning_day", "time_period"]
BINARY_COLS = ["is_rain", "is_snow"]


def load_dataset(path) -> pd.DataFrame:
    """Read a project CSV and rename its columns to the canonical names."""
    return pd.read_csv(path).rename(columns=RENAME_MAP)


class ManualStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize numeric features with a hand-written (x - mean) / std.

    Parameters (``mean_``, ``scale_``) are learned in ``fit`` on the train fold
    only, so there is no leakage to validation/test. Equivalent to sklearn's
    StandardScaler (population std, ddof=0), but written out explicitly per the
    project's scaling convention. The preceding imputer guarantees no NaN here.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)  # ddof=0, как в StandardScaler
        std[std == 0] = 1.0  # константный столбец - не делим на ноль
        self.scale_ = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_


class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn implicit weather/time signals into explicit features.

    The transformer is stateless (``fit`` only validates and returns ``self``)
    so it never leaks information from validation/test folds. It produces, on the
    BikeSouth hypothesis that demand depends on *combinations* of factors:

    - ``time_period``    - single categorical rebuilt from the four one-hot
      columns; the all-False case is the implicit "Daytime" period.
    - ``temp_solar``     - temperature x solar radiation; separates "+28 under a
      blazing sun" from "+28 after the rain".
    - ``temp_dewpoint_gap`` - temperature minus dew point; a small gap means
      muggy air, a large gap means dry/comfortable air.
    - ``is_rain`` / ``is_snow`` - binary "any precipitation" flags; precipitation
      is rare but suppresses demand sharply.

    The four raw time one-hot columns are dropped because ``time_period`` carries
    the same information without the redundant block.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        conditions = [
            X["time_period_night"].astype(bool),
            X["time_period_morning"].astype(bool),
            X["time_period_evening"].astype(bool),
            X["time_period_late_evening"].astype(bool),
        ]
        X["time_period"] = np.select(
            conditions,
            ["Night", "Morning", "Evening", "Late Evening"],
            default="Daytime",
        )
        X["temp_solar"] = X["temperature"] * X["solar_radiation_mjm2"]
        X["temp_dewpoint_gap"] = X["temperature"] - X["dew_point_temperature"]
        # NaN precipitation is treated as "no precipitation" (a per-row rule, not
        # a train statistic); the continuous columns are still median-imputed
        # downstream inside the ColumnTransformer.
        X["is_rain"] = (X["rainfallmm"].fillna(0) > 0).astype(int)
        X["is_snow"] = (X["snowfall_cm"].fillna(0) > 0).astype(int)
        return X.drop(columns=TIME_BOOL_COLS)


def build_preprocessor():
    """ColumnTransformer: median-impute + manual-scale numerics, impute + OHE categoricals."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", ManualStandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
    )


def feature_names(preprocessor):
    """Output feature names of a fitted build_preprocessor(), built by hand.

    Avoids ``get_feature_names_out`` (sklearn >= 1.0); works on sklearn 0.24.2.
    Order matches the ColumnTransformer: numeric -> one-hot categorical -> binary.
    """
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = []
    for col, cats in zip(CATEGORICAL_COLS, ohe.categories_):
        cat_names += [f"{col}_{c}" for c in cats]
    return list(NUMERIC_COLS) + cat_names + list(BINARY_COLS)
