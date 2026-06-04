"""Артефакт для внедрения: сборка признаков + ручной стандартизатор.

Этот модуль импортируется при загрузке пайплайна joblib-ом, поэтому он должен
существовать в путях импорта на проде.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DAYTIME_ORDER = ["утро", "день", "вечер", "ночь"]
DAYTIME_DROP = "ночь"
ALL_CATEGORIES = [f"Category {i:02d}" for i in range(1, 21)]
CATEGORY_DROP = "Category 20"


class ManualStandardScaler(BaseEstimator, TransformerMixin):
    """Стандартизация (x - mean) / std вручную; параметры берутся только из train."""

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = X_arr.mean(axis=0)
        scale = X_arr.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        self.n_features_in_ = X_arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def build_features(visits, ads, surf, cloud):
    """По сырым датафреймам возвращает признаки пользователей, индекс — user_id.

    Возвращает n_events, sessions_per_day, 3 доли времени суток, 19 долей категорий
    плюс 4 категориальных колонки (ads/surf/cloud/peak_daytime).
    """
    v = visits.copy()
    v["date"] = pd.to_datetime(v["date"])

    base = v.groupby("user_id").agg(
        n_events=("session_id", "size"),
        n_sessions=("session_id", "nunique"),
        n_active_days=("date", "nunique"),
    )
    base["sessions_per_day"] = base["n_sessions"] / base["n_active_days"].clip(lower=1)
    base = base.drop(columns=["n_sessions", "n_active_days"])

    daytime_full = (
        pd.crosstab(v["user_id"], v["daytime"], normalize="index")
        .reindex(columns=DAYTIME_ORDER, fill_value=0.0)
    )
    peak_daytime = daytime_full.idxmax(axis=1).rename("peak_daytime")
    daytime_share = daytime_full.drop(columns=DAYTIME_DROP).add_prefix("daytime_")

    cat_share = (
        pd.crosstab(v["user_id"], v["website_category"], normalize="index")
        .reindex(columns=ALL_CATEGORIES, fill_value=0.0)
        .drop(columns=CATEGORY_DROP)
        .add_prefix("cat_")
    )

    features = base.join(daytime_share, how="left").join(cat_share, how="left")
    features = features.fillna(0.0)

    def _lookup(df, key, value):
        if df is None or df.empty:
            return pd.Series(index=features.index, dtype=object).fillna("unknown")
        s = df.drop_duplicates(subset=key).set_index(key)[value]
        return features.index.to_series().map(s).astype(object).fillna("unknown")

    features["ads_activity"] = _lookup(ads,   "user_id", "ads_activity")
    features["surf_depth"]   = _lookup(surf,  "user_id", "surf_depth")
    features["cloud_usage"]  = _lookup(cloud, "user_id", "cloud_usage").astype(str)
    features["peak_daytime"] = features.index.to_series().map(peak_daytime)
    return features
