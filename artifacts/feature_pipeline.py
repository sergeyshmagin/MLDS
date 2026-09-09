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

# Маркер спринта: в прошлых спринтах модуль назывался так же, и тетрадь может
# случайно импортировать чужой файл из домашней папки. Тетрадь проверяет это значение.
SPRINT = "SPRINT_18"

RANDOM_STATE = 42

# Экономические параметры из ТЗ, рубли.
AVG_REV = 64_500        # средний доход от сдачи одного номера
COST_FP = 7_000         # цена ложного срабатывания (компенсация + админ + маркетинг)
LOST_REV = 64_500       # упущенный доход от незамеченной отмены
PER_REBOOKING = 45_000  # доход от повторной сдачи вовремя замеченного номера

TARGET = "is_cancelled"
POSITIVE_STATUS = "отказ_брони"

BASE_NUMERIC_COLS = [
    "days_until_checkin",
    "total_nights",
    "weekend_share",
    "guests_total",
    "child_count",
    "price_per_night",
    "previous_cancellations",
    "previous_no_shows",
    "customer_loyalty",
    "customer_special_requests",
    "returning_customer",
    "parking_included",
    "checkin_month",
    "checkin_dayofweek",
    "booking_month",
]

# booking_year в признаки не входит: разбиение хронологическое, поэтому на калибровочной
# и тестовой выборках встречаются годы, которых не было в обучении, и модель вынуждена
# экстраполировать. Год оформления - свойство периода выгрузки, а не самой брони.
EXCLUDED_COLS = ["booking_year"]

# Признаки из отзывов (stay_rating, review_lag_days и TF-IDF по review_text) в модель
# не входят. После объединения по дате они относятся к постороннему гостю, а не к клиенту
# текущей брони: связь по клиенту невозможна без customer_id, известного в момент
# оформления. Объединение и векторизация остаются в тетради как исследовательский этап.
REVIEW_COLS = ["stay_rating", "review_lag_days", "review_text"]

# Признаки спроса: сколько броней на ту же дату заезда уже было оформлено к моменту
# текущей брони и насколько активно бронировали в предыдущие 30 дней. Обе величины
# известны отелю в момент прогноза и считаются только по ранее оформленным броням.
DEMAND_COLS = ["same_checkin_prior_bookings", "bookings_prev_30d"]

CATEGORICAL_COLS = [
    "sales_channel",
    "meal_plan",
    "room_type",
    "season_group",
    "lead_time_group",
]

TEXT_COL = "review_text"

# Месяцы заезда, объединённые в группы для категоризации сезонности.
SUMMER_WINTER_MONTHS = (6, 7, 8, 12)
SHOULDER_MONTHS = (2, 3, 10, 11)


def numeric_cols(with_demand=False):
    """Список числовых признаков; признаки спроса подключаются флагом."""
    return BASE_NUMERIC_COLS + (DEMAND_COLS if with_demand else [])


def feature_cols(with_demand=False):
    return numeric_cols(with_demand) + CATEGORICAL_COLS


# Совместимость с сохранёнными артефактами и короткими обращениями из тетради.
NUMERIC_COLS = BASE_NUMERIC_COLS
FEATURE_COLS = feature_cols(False)


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
    # Нулевая стоимость брони невозможна: минимальная ненулевая цена 6 700 руб.
    # Это некорректные строки, поэтому они удаляются, а не восстанавливаются импутацией -
    # придумывать стоимость там, где её не было, значит подменять данные.
    out = out[out["booking_value"] > 0].reset_index(drop=True)
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
    merge_asof с direction='backward' и allow_exact_matches=False берёт
    последний отзыв, оставленный строго ДО даты оформления брони.
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
        allow_exact_matches=False,
    )
    merged["review_lag_days"] = (
        merged["booking_date"] - merged["review_date"]
    ).dt.days
    merged["review_text"] = merged["review_text"].fillna("")
    return merged


def add_demand_features(df):
    """Считает признаки спроса только по ранее оформленным броням.

    Строки упорядочены по дате оформления, поэтому кумулятивный счётчик по дате
    заезда видит лишь брони, оформленные раньше текущей. Спрос за предыдущие
    30 дней берётся со сдвигом на день, чтобы текущий день в него не попал.
    """
    out = df.sort_values("booking_date").reset_index(drop=True).copy()
    checkin_date = out["booking_date"] + pd.to_timedelta(
        out["days_until_checkin"], unit="D"
    )
    out["same_checkin_prior_bookings"] = out.groupby(checkin_date).cumcount()

    daily = out.groupby("booking_date").size().sort_index()
    rolling_30d = daily.rolling("30D").sum() - daily
    out["bookings_prev_30d"] = out["booking_date"].map(rolling_30d).fillna(0)
    return out


def build_features(df, with_demand=False):
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

    # Лояльность клиента из ТЗ: доля заездов среди прошлых бронирований клиента.
    # Сглаживание Лапласа даёт новым клиентам нейтральные 0.5 вместо нуля.
    out["customer_loyalty"] = (out["previous_no_shows"] + 1) / (
        out["previous_no_shows"] + out["previous_cancellations"] + 2
    )

    out["season_group"] = np.select(
        [
            out["checkin_month"].isin(SUMMER_WINTER_MONTHS),
            out["checkin_month"].isin(SHOULDER_MONTHS),
        ],
        ["лето_и_декабрь", "межсезонье"],
        default="прочие_месяцы",
    )
    out["lead_time_group"] = pd.cut(
        out["days_until_checkin"],
        bins=[-np.inf, 30, 90, 180, np.inf],
        labels=["до_месяца", "1-3_месяца", "3-6_месяцев", "более_полугода"],
    ).astype(str)
    return out[feature_cols(with_demand)]


def make_preprocessor(with_demand=False):
    """Препроцессор: импьютер по медиане и one-hot по категориям.

    Все статистики (медианы, словарь категорий) считаются внутри fit, поэтому
    препроцессор обучается только на обучающей части - в том числе внутри
    каждого фолда кросс-валидации.
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", numeric, numeric_cols(with_demand)),
            ("cat", categorical, CATEGORICAL_COLS),
        ],
        sparse_threshold=0.0,
    )


def make_review_vectorizer(max_features=60, min_df=50):
    """Векторизатор текстов отзывов для исследовательского раздела тетради.

    В модель эти признаки не входят: после объединения по дате отзыв относится
    к постороннему гостю.
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        lowercase=True,
        token_pattern=r"(?u)\b[а-яёa-z]{3,}\b",
    )


def get_feature_names(preprocessor, with_demand=False):
    """Имена колонок после препроцессора (в sklearn 0.24 нет get_feature_names_out)."""
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names(CATEGORICAL_COLS)
    )
    return numeric_cols(with_demand) + cat_names


def collapse_onehot(shap_values, feature_names, with_demand=False):
    """Сворачивает вклады one-hot дамми обратно к исходным признакам.

    Вклады дамми одного признака складываются ВНУТРИ строки и только потом
    берётся модуль: сумма средних модулей по колонкам завышала бы вклад
    категориальных признаков.
    """
    columns = list(feature_names)
    groups = {}
    for name in numeric_cols(with_demand):
        groups[name] = [columns.index(name)]
    for cat in CATEGORICAL_COLS:
        groups[cat] = [i for i, name in enumerate(columns) if name.startswith(cat + "_")]
    collapsed = np.column_stack(
        [shap_values[:, idx].sum(axis=1) for idx in groups.values()]
    )
    return collapsed, list(groups.keys())


class CalibratedBookingModel:
    """Препроцессор, базовая модель и калибратор, обученный на временных OOF-прогнозах.

    Класс живёт в модуле, а не в тетради, чтобы joblib.load работал без патчей
    ``__module__``. Интерфейс повторяет sklearn: ``predict_proba`` возвращает
    две колонки вероятностей.
    """

    def __init__(self, preprocessor, model, calibrator, with_demand=False):
        self.preprocessor = preprocessor
        self.model = model
        self.calibrator = calibrator
        self.with_demand = with_demand

    def predict_proba(self, X):
        raw = self.model.predict_proba(self.preprocessor.transform(X))[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


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


def business_metrics(y_true, y_pred, rooms_before=None, rooms_after=None):
    """Бизнес-показатели до и после внедрения модели.

    ``rooms_before`` и ``rooms_after`` - число доступных номеров, среднее за
    период до тестирования и за период тестирования. Если не заданы, знаменателем
    служит число броней выборки (одной броне соответствует один номер).

    Занятые номера до внедрения - все брони, по которым гость приехал: TN + FP.
    После внедрения занятыми считаются только TN: по броням FP номер снят с продажи
    и передан другому гостю, а по TP факта нового заселения в данных нет - доход от
    возможной повторной продажи отражён отдельно, через PerRebooking в формуле IR.
    Рядом возвращается справочный вариант, в котором перепроданные номера (TP) всё же
    засчитываются занятыми: он показывает, каким был бы результат, если бы факт
    повторного заселения фиксировался в данных.
    """
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    n = len(y_true)
    rooms_before = n if rooms_before is None else rooms_before
    rooms_after = n if rooms_after is None else rooms_after

    ir_before = revenue_before(y_true)
    ir_after = revenue_after(y_true, y_pred)

    cancel_rate_before = y_true.sum() / n
    cancel_rate_after = fn / n
    occupancy_before = (tn + fp) / rooms_before
    occupancy_after = tn / rooms_after
    occupancy_after_with_tp = (tn + tp) / rooms_after

    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "IR_before": ir_before,
        "IR_after": ir_after,
        "IR": ir_after - ir_before,
        "IR_rel_%": (ir_after - ir_before) / ir_before * 100,
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
        "occupancy_after_with_TP_%": occupancy_after_with_tp * 100,
        "occupancy_dynamics_with_TP_%": (occupancy_before - occupancy_after_with_tp)
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
