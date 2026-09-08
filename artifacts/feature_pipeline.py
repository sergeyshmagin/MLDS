"""Подготовка данных и признаков для модели отмены бронирования UrbanStay.

Модуль - источник истины для очистки данных, объединения таблиц, генерации
признаков и расчёта бизнес-метрики Incremental Revenue. Тетрадь и артефакт
модели импортируют функции отсюда, поэтому классы имеют канонический
``__module__`` и joblib.load работает без патчей.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# Экономические параметры из ТЗ, рубли.
AVG_REV = 64_500        # средний доход от сдачи одного номера
COST_FP = 7_000         # цена ложного срабатывания (компенсация + админ + маркетинг)
LOST_REV = 64_500       # упущенный доход от незамеченной отмены
PER_REBOOKING = 45_000  # доход от повторной сдачи вовремя замеченного номера

TARGET = "is_cancelled"
POSITIVE_STATUS = "отказ_брони"

NUMERIC_COLS = [
    "days_until_checkin",
    "total_nights",
    "weekend_share",
    "guests_total",
    "child_count",
    "price_per_night",
    "previous_cancellations",
    "previous_no_shows",
    "customer_special_requests",
    "returning_customer",
    "parking_included",
    "checkin_month",
    "checkin_dayofweek",
    "booking_month",
    "booking_year",
    "stay_rating",
    "review_lag_days",
]

CATEGORICAL_COLS = [
    "sales_channel",
    "meal_plan",
    "room_type",
    "season_group",
    "lead_time_group",
]

TEXT_COL = "review_text"

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]

# Месяцы заезда с высоким и низким спросом: лето и новогодние праздники против
# межсезонья. Используется для категоризации признака сезонности.
HIGH_SEASON_MONTHS = (6, 7, 8, 12)
LOW_SEASON_MONTHS = (2, 3, 10, 11)


def clean_bookings(df):
    """Очистка таблицы бронирований: дубликаты, типы, ошибки ввода."""
    out = df.drop_duplicates().reset_index(drop=True).copy()
    out["booking_date"] = pd.to_datetime(out["booking_date"])
    # 100/200/300 взрослых в номере физически невозможны: это те же 1/2/3
    # с лишними нулями при вводе, доля таких строк около 6%.
    mask_x100 = out["adult_count"] >= 100
    out.loc[mask_x100, "adult_count"] = out.loc[mask_x100, "adult_count"] // 100
    # Опечатка в справочнике питания.
    out["meal_plan"] = out["meal_plan"].replace({"не выбрант": "не выбран"})
    # Нулевая стоимость брони - это незаполненное поле, а не бесплатный номер:
    # минимальная ненулевая цена 6 700 руб. Ноль переводим в пропуск, чтобы его
    # заполнил импьютер по медиане обучающей выборки.
    out["booking_value"] = out["booking_value"].replace({0.0: np.nan})
    for col in ("returning_customer", "parking_included"):
        out[col] = out[col].astype(int)
    out[TARGET] = (out["booking_status"] == POSITIVE_STATUS).astype(int)
    return out


def clean_reviews(df):
    """Очистка таблицы отзывов."""
    out = df.drop_duplicates().reset_index(drop=True).copy()
    out["review_date"] = pd.to_datetime(out["review_date"])
    out["review_text"] = out["review_text"].fillna("").str.strip()
    return out


def join_reviews(bookings, reviews):
    """Присоединяет к каждой брони ближайший предшествующий отзыв по дате.

    Ключа клиента в таблице бронирований нет, а восстановление его через
    booking_id таблицы отзывов даёт утечку целевой переменной (отзыв есть
    только у состоявшихся броней). Поэтому связь строится только по времени:
    merge_asof с direction='backward' берёт последний отзыв, который был
    известен отелю на дату оформления брони.
    """
    left = bookings.sort_values("booking_date").reset_index(drop=True)
    right = (
        reviews[["review_date", "stay_rating", "review_text"]]
        .sort_values("review_date")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="booking_date",
        right_on="review_date",
        direction="backward",
    )
    merged["review_lag_days"] = (
        merged["booking_date"] - merged["review_date"]
    ).dt.days
    merged["review_text"] = merged["review_text"].fillna("")
    return merged


def build_features(df):
    """Собирает матрицу признаков из объединённой таблицы."""
    out = df.copy()
    checkin_date = out["booking_date"] + pd.to_timedelta(
        out["days_until_checkin"], unit="D"
    )
    out["checkin_month"] = checkin_date.dt.month
    out["checkin_dayofweek"] = checkin_date.dt.dayofweek
    out["booking_month"] = out["booking_date"].dt.month
    out["booking_year"] = out["booking_date"].dt.year

    out["total_nights"] = out["weekday_nights"] + out["weekend_nights"]
    out["weekend_share"] = out["weekend_nights"] / out["total_nights"]
    out["guests_total"] = out["adult_count"] + out["child_count"]
    out["price_per_night"] = out["booking_value"] / out["total_nights"]

    out["season_group"] = np.select(
        [
            out["checkin_month"].isin(HIGH_SEASON_MONTHS),
            out["checkin_month"].isin(LOW_SEASON_MONTHS),
        ],
        ["высокий_сезон", "низкий_сезон"],
        default="средний_сезон",
    )
    out["lead_time_group"] = pd.cut(
        out["days_until_checkin"],
        bins=[-np.inf, 30, 90, 180, np.inf],
        labels=["до_месяца", "1-3_месяца", "3-6_месяцев", "более_полугода"],
    ).astype(str)
    return out[FEATURE_COLS]


def make_preprocessor(tfidf_max_features=60, tfidf_min_df=50):
    """Препроцессор: импьютер по медиане, one-hot и TF-IDF по текстам отзывов.

    Все статистики (медианы, словарь категорий, словарь TF-IDF) считаются
    внутри fit, поэтому препроцессор обучается только на train.
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse=False)
    text = TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
        lowercase=True,
        token_pattern=r"(?u)\b[а-яёa-z]{3,}\b",
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
            ("txt", text, TEXT_COL),
        ],
        sparse_threshold=0.0,
    )


def get_feature_names(preprocessor):
    """Имена колонок после препроцессора (в sklearn 0.24 нет get_feature_names_out)."""
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names(CATEGORICAL_COLS)
    )
    txt_names = [
        "tfidf_" + w
        for w in preprocessor.named_transformers_["txt"].get_feature_names()
    ]
    return NUMERIC_COLS + cat_names + txt_names


def confusion_counts(y_true, y_pred):
    """TN, FP, FN, TP без sklearn - чтобы формулы IR читались напрямую."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def revenue_before(y_true):
    """IR до внедрения модели: доход от состоявшихся броней минус потери от отмен."""
    y_true = np.asarray(y_true).astype(int)
    total_cancellations = int(y_true.sum())
    total_success = int(len(y_true) - total_cancellations)
    return total_success * AVG_REV - total_cancellations * LOST_REV


def revenue_after(y_true, y_pred):
    """IR после внедрения модели по матрице ошибок."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return tp * PER_REBOOKING + tn * AVG_REV - fp * COST_FP - fn * LOST_REV


def business_metrics(y_true, y_pred):
    """Полный набор бизнес-показателей до и после внедрения модели.

    Загрузка после внедрения считается консервативно: занятыми считаем номера
    вовремя перепроданных отмен (TP) и состоявшихся броней без вмешательства
    модели (TN). Номера из FP отель тоже сдаёт, но исходный гость приезжает,
    поэтому в загрузку они не засчитываются, а цена конфликта уже учтена
    в CostFP.
    """
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    n = len(y_true)

    ir_before = revenue_before(y_true)
    ir_after = revenue_after(y_true, y_pred)

    cancel_rate_before = y_true.sum() / n
    cancel_rate_after = fn / n
    occupancy_before = (n - y_true.sum()) / n
    occupancy_after = (tp + tn) / n

    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "IR_before": ir_before,
        "IR_after": ir_after,
        "IR": ir_after - ir_before,
        "IR_rel_%": (ir_after - ir_before) / abs(ir_before) * 100,
        "cancel_before_%": cancel_rate_before * 100,
        "cancel_after_%": cancel_rate_after * 100,
        "cancel_dynamics_%": (cancel_rate_before - cancel_rate_after)
        / cancel_rate_before
        * 100,
        "occupancy_before_%": occupancy_before * 100,
        "occupancy_after_%": occupancy_after * 100,
        "occupancy_dynamics_%": (occupancy_before - occupancy_after)
        / occupancy_before
        * 100,
    }


def threshold_scan(y_true, proba, thresholds):
    """Таблица бизнес-показателей по сетке порогов классификации."""
    rows = []
    for thr in thresholds:
        metrics = business_metrics(y_true, (proba >= thr).astype(int))
        metrics["threshold"] = thr
        rows.append(metrics)
    cols = [
        "threshold", "TP", "FP", "FN", "TN", "IR_after", "IR", "IR_rel_%",
        "cancel_after_%", "occupancy_after_%",
    ]
    return pd.DataFrame(rows)[cols]


def break_even_threshold():
    """Порог безубыточности, следующий прямо из матрицы стоимостей.

    Для брони с вероятностью отмены p прогноз «отмена» выгоднее, если
    p * (PerRebooking + LostRev) > (1 - p) * (AvgRev + CostFP).
    """
    gain_positive = PER_REBOOKING + LOST_REV
    loss_negative = AVG_REV + COST_FP
    return loss_negative / (gain_positive + loss_negative)
