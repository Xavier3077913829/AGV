from __future__ import annotations

import warnings
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_LABELS = {
    "moving_average": "移动平均",
    "holt_winters": "Holt-Winters",
    "arima": "ARIMA",
    "lightgbm": "LightGBM（日历+滞后）",
}

ALL_MODELS = tuple(MODEL_LABELS)
RANDOM_SEED = 42


class DemandForecastError(ValueError):
    """订单预测输入或计算异常。"""


def generate_order_series(days: int = 240, seed: int = RANDOM_SEED) -> list[dict]:
    """生成具有趋势、周周期和随机扰动的订单序列。"""
    if days < 30:
        days = 30

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    index = np.arange(days)
    trend = 100.0 + 0.08 * index
    weekly = 12.0 * np.sin(2.0 * np.pi * index / 7.0)
    monthly = 7.0 * np.sin(2.0 * np.pi * index / 30.0)
    weekend = np.where(dates.dayofweek >= 5, -9.0, 0.0)
    noise = rng.normal(0.0, 5.0, size=days)
    values = np.maximum(trend + weekly + monthly + weekend + noise, 1.0)

    return [
        {
            "date": dates[position].strftime("%Y-%m-%d"),
            "order_count": int(round(values[position])),
        }
        for position in range(days)
    ]


SERIES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "forecast"

SERIES_SOURCES = {
    "online_retail_daily": "UCI Online Retail 真实交易按天聚合（2010-12 ~ 2011-12，541,909 条交易）",
}



# ---- 列名别名：兼容老师给的 date/store/item/sales 格式 ----
_DATE_ALIASES = ("date", "日期", "dt", "day", "datetime", "time")
_VALUE_ALIASES = (
    "order_count", "sales", "sales_qty", "qty", "quantity", "amount",
    "order_qty", "volume", "销量", "订单量", "销量(件)",
)
_GROUP_ALIASES = ("store", "item", "sku", "product", "store_id", "item_id", "门店", "商品")


def standardize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把任意订单 CSV 归一到 date + order_count。

    支持：
      - date, order_count                       （原格式）
      - date, sales                             （老师给的数据）
      - date, store, item, sales                （多序列 → 按 date 汇总求和）
    返回 (归一化后的 frame, 说明信息)。
    """
    frame = frame.copy()
    lower = {str(c).strip().lower(): c for c in frame.columns}

    def pick(aliases):
        for a in aliases:
            if a in lower:
                return lower[a]
        return None

    date_col = pick(_DATE_ALIASES)
    value_col = pick(_VALUE_ALIASES)
    if value_col is None:
        # 退一步：任意数值列里找名字含 count/sales/qty 的
        for c in frame.columns:
            name = str(c).lower()
            if any(k in name for k in ("count", "sales", "qty", "quantity", "amount")):
                value_col = c
                break
    if date_col is None or value_col is None:
        raise DemandForecastError(
            "订单数据需要日期列与数值列：支持 date + order_count，也支持 date + sales（含 store/item 时按天汇总）。"
        )

    group_cols = [lower[a] for a in _GROUP_ALIASES if a in lower]
    work = frame[[date_col, value_col] + group_cols].copy()
    work = work.rename(columns={date_col: "date", value_col: "order_count"})
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["order_count"] = pd.to_numeric(work["order_count"], errors="coerce")
    work = work.dropna(subset=["date", "order_count"])

    info = {"value_column": str(value_col), "aggregated": False, "groups": 1}
    if group_cols:
        n_groups = int(work.groupby(group_cols).ngroups) if len(work) else 0
        work = (
            work.groupby("date", as_index=False)["order_count"]
            .sum()
            .sort_values("date")
        )
        info.update({"aggregated": True, "groups": n_groups,
                     "note": f"检测到 {group_cols} → 已按日期汇总 {n_groups} 条序列"})
    return work, info


def available_series() -> list[dict]:
    """列出 datasets/forecast 下可直接预测的真实订单序列。"""
    if not SERIES_DIR.is_dir():
        return []
    series_list = []
    for path in sorted(SERIES_DIR.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - 坏文件跳过，不影响其他序列
            continue
        if not {"date", "order_count"}.issubset(frame.columns) or frame.empty:
            continue
        values = pd.to_numeric(frame["order_count"], errors="coerce")
        series_list.append(
            {
                "name": path.stem,
                "file": path.name,
                "points": int(len(frame)),
                "start": str(frame["date"].iloc[0]),
                "end": str(frame["date"].iloc[-1]),
                "total": float(values.sum()),
                "mean": round(float(values.mean()), 2),
                "source": SERIES_SOURCES.get(path.stem, "本地数据集"),
            }
        )
    return series_list


def _resolve_series_path(name: str) -> Path:
    """把序列名解析成 datasets/forecast 下的 CSV 路径，拒绝越出目录的名字。"""
    stem = str(name or "").strip()
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    path = (SERIES_DIR / f"{stem}.csv").resolve()
    if path.parent != SERIES_DIR.resolve() or not path.is_file():
        raise DemandForecastError(f"订单序列不存在：{name}")
    return path


def load_series(name: str) -> list[dict]:
    """按名称读取真实订单序列，返回可直接交给 forecast_demand 的记录列表。"""
    frame = pd.read_csv(_resolve_series_path(name))
    if not {"date", "order_count"}.issubset(frame.columns):
        raise DemandForecastError("订单序列必须包含 date 与 order_count 字段。")
    normalized = _normalize_history(frame)
    return [
        {"date": row.date.strftime("%Y-%m-%d"), "order_count": round(float(row.order_count), 2)}
        for row in normalized.itertuples(index=False)
    ]


def _infer_frequency(frame: pd.DataFrame) -> tuple[str, int, int]:
    """按相邻日期间隔推断 (pandas 频率, 季节周期, 移动平均窗口)。

    日数据 → 周周期 7；周数据 → 年周期 52（样本不足退回 4）；月数据 → 年周期 12。
    """
    if len(frame) < 3:
        return "D", 7, 7
    step = float(frame["date"].diff().dt.days.dropna().median())
    if step <= 2:
        return "D", 7, 7
    if step <= 10:
        return "W", (52 if len(frame) >= 104 else 4), 4
    if 25 <= step <= 35:
        return "MS", (12 if len(frame) >= 24 else 4), 3
    if 80 <= step <= 100:
        return "QS", 4, 4
    return "D", 7, 7


def _normalize_history(history: Iterable[dict] | pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame(list(history))
    if frame.empty:
        raise DemandForecastError("历史订单数据不能为空。")
    # 先做列名标准化（兼容 date/sales、store/item 多序列），再校验
    frame, _info = standardize_frame(frame)
    if meta is not None:
        meta.update(_info)
    normalized = frame[["date", "order_count"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["order_count"] = pd.to_numeric(normalized["order_count"], errors="coerce")
    normalized = normalized.dropna().sort_values("date").drop_duplicates("date")
    if len(normalized) < 30:
        raise DemandForecastError("至少需要 30 条有效历史订单数据。")
    return normalized.reset_index(drop=True)


def _moving_average_forecast(
    values: np.ndarray, horizon: int, window: int | None = None
) -> np.ndarray:
    history = [float(value) for value in values]
    window = min(window or 7, len(history))
    result = []
    for _ in range(horizon):
        forecast = float(np.mean(history[-window:]))
        result.append(forecast)
        history.append(forecast)
    return np.asarray(result, dtype=float)


def _holt_winters_forecast(
    values: np.ndarray, horizon: int, seasonal_period: int = 7
) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    period = max(2, int(seasonal_period))
    use_seasonal = len(series) >= 2 * period
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=period if use_seasonal else None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(horizon)
    return np.asarray(forecast, dtype=float)


def _arima_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            values,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon).predicted_mean
    return np.asarray(forecast, dtype=float)


def _lightgbm_vector(history: np.ndarray, when: pd.Timestamp) -> list[float]:
    """单个时间点的特征向量（训练与推理共用，保证口径一致）。

    特征：星期/月份/年内第几天/是否周末 + 周与年周期的正弦余弦（不含线性趋势，
    避免树模型把“序号”当趋势外推）+ 多阶滞后 + 排除当天的滚动统计。
    """
    hist = np.asarray(history, dtype=float)
    n = len(hist)
    dow = int(when.dayofweek)
    doy = int(when.dayofyear)
    feats = [dow, int(when.month), doy, int(dow >= 5)]
    for k in (1, 2, 3):
        feats += [float(np.sin(2 * np.pi * k * dow / 7)), float(np.cos(2 * np.pi * k * dow / 7))]
    for k in (1, 2, 3, 4, 5, 6):
        feats += [float(np.sin(2 * np.pi * k * doy / 365.25)),
                  float(np.cos(2 * np.pi * k * doy / 365.25))]
    for lag in (1, 2, 3, 7, 14, 21, 28, 364):
        feats.append(float(hist[n - lag]) if n >= lag else float(np.mean(hist)))
    for w in (7, 28):
        seg = hist[max(0, n - w):n]
        feats += [float(seg.mean()), float(seg.std()), float(seg.min()), float(seg.max())]
    return feats


def _lightgbm_forecast(values: np.ndarray, horizon: int, dates=None) -> np.ndarray:
    """LightGBM 递推多步预测（每预测一天把预测值并入历史）。"""
    import lightgbm as lgb

    values = np.asarray(values, dtype=float)
    if dates is None:
        dates = pd.date_range("2000-01-01", periods=len(values), freq="D")
    dates = pd.DatetimeIndex(dates)
    start = min(365, max(0, len(values) // 3))
    if len(values) - start < 60:
        raise ValueError("LightGBM 需要至少约 400 天历史数据")

    X = np.asarray([_lightgbm_vector(values[:i], dates[i]) for i in range(start, len(values))], float)
    y = values[start:]
    model = lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.04, num_leaves=31, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, verbose=-1, n_jobs=4,
    )
    model.fit(X, y)

    step_days = float(pd.Series(dates).diff().dt.days.dropna().median() or 1)
    freq = "MS" if step_days >= 25 else ("W" if step_days >= 6 else "D")
    future_dates = pd.date_range(dates[-1], periods=horizon + 1, freq=freq)[1:]

    history = list(values)
    preds = []
    for fd in future_dates:
        feat = np.asarray([_lightgbm_vector(np.asarray(history, float), fd)], float)
        v = max(float(model.predict(feat)[0]), 0.0)
        preds.append(v)
        history.append(v)
    return np.asarray(preds, dtype=float)


def _fit_forecast(
    model_name: str,
    values: np.ndarray,
    horizon: int,
    seasonal_period: int = 7,
    ma_window: int | None = None,
    dates=None,
) -> np.ndarray:
    if model_name == "moving_average":
        return _moving_average_forecast(values, horizon, ma_window)
    if model_name == "holt_winters":
        return _holt_winters_forecast(values, horizon, seasonal_period)
    if model_name == "arima":
        return _arima_forecast(values, horizon)
    if model_name == "lightgbm":
        return _lightgbm_forecast(values, horizon, dates)
    raise DemandForecastError(f"未知预测模型：{model_name}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(sqrt(np.mean(error**2)))
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if np.any(nonzero)
        else 0.0
    )
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def forecast_demand(
    history: Iterable[dict] | pd.DataFrame,
    horizon: int = 14,
    model: str = "auto",
) -> dict:
    """回测候选模型，选择误差最小者并预测未来订单。"""
    _meta: dict = {}
    frame = _normalize_history(history, meta=_meta)
    horizon = int(horizon)
    if horizon < 1 or horizon > 120:
        raise DemandForecastError("预测期数必须在 1 到 120 之间。")

    model = (model or "auto").strip().lower()
    if model == "holt-winters":
        model = "holt_winters"
    candidate_models = ALL_MODELS if model == "auto" else (model,)
    if any(name not in ALL_MODELS for name in candidate_models):
        raise DemandForecastError(
            "model 只能是 auto、lightgbm、moving_average、holt_winters 或 arima。"
        )

    values = frame["order_count"].to_numpy(dtype=float)
    frequency, seasonal_period, ma_window = _infer_frequency(frame)
    validation_horizon = min(max(horizon, 7), max(7, len(values) // 5))
    train_values = values[:-validation_horizon]
    actual_values = values[-validation_horizon:]
    # LightGBM 的日历特征（周几/月份/年内第几天）必须用真实日期，
    # 否则会退化成从 2000-01-01 起的假日期，周频/月频数据全部错位。
    all_dates = pd.DatetimeIndex(frame["date"])
    train_dates = all_dates[:-validation_horizon]
    if len(train_values) < 20:
        raise DemandForecastError("回测训练数据不足，请增加历史数据量。")

    candidates = {}
    for candidate in candidate_models:
        try:
            predicted = _fit_forecast(
                candidate, train_values, validation_horizon,
                seasonal_period, ma_window, dates=train_dates,
            )
            if len(predicted) != validation_horizon or not np.all(np.isfinite(predicted)):
                raise ValueError("forecast contains invalid values")
            candidates[candidate] = _metrics(actual_values, predicted)
        except Exception as exc:
            candidates[candidate] = {
                "mae": None,
                "rmse": None,
                "mape": None,
                "error": str(exc),
            }

    valid_candidates = {
        name: metrics
        for name, metrics in candidates.items()
        if metrics["rmse"] is not None
    }
    if not valid_candidates:
        raise DemandForecastError("候选模型全部训练失败，请更换历史数据。")

    selected_model = min(
        valid_candidates,
        key=lambda name: (
            valid_candidates[name]["rmse"],
            valid_candidates[name]["mape"],
        ),
    )
    future_values = _fit_forecast(
        selected_model, values, horizon, seasonal_period, ma_window, dates=all_dates
    )
    future_values = np.maximum(future_values, 0.0)

    last_date = frame["date"].iloc[-1]
    # 周频的 "W" 锚定在周日，直接用它生成会把未来日期挪到周日（和原始星期几不一致），
    # 这里改成「距上一期 7 天」，保持序列自身的周期位置。
    date_freq = "7D" if frequency == "W" else frequency
    future_dates = pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=date_freq,
    )[1:]

    history_payload = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "order_count": round(float(row.order_count), 2),
        }
        for row in frame.itertuples(index=False)
    ]
    forecast_payload = [
        {
            "date": future_dates[index].strftime("%Y-%m-%d"),
            "order_count": round(float(future_values[index]), 2),
        }
        for index in range(horizon)
    ]
    selected_metrics = valid_candidates[selected_model]

    return {
        "selected_model": selected_model,
        "selected_model_label": MODEL_LABELS[selected_model],
        "model_requested": model,
        "horizon": horizon,
        "history_count": len(frame),
        "frequency": frequency,
        "seasonal_period": seasonal_period,
        "ma_window": ma_window,
        "predicted_total": round(float(np.sum(future_values)), 2),
        "metrics": selected_metrics,
        "candidate_metrics": candidates,
        "history": history_payload,
        "forecast": forecast_payload,
        "aggregation": dict(_meta),
    }
