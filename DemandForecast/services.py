from __future__ import annotations

import warnings
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_LABELS = {
    "moving_average": "移动平均",
    "holt_winters": "Holt-Winters",
    "arima": "ARIMA",
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


def _normalize_history(history: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame(list(history))
    if frame.empty:
        raise DemandForecastError("历史订单数据不能为空。")
    if "date" not in frame.columns or "order_count" not in frame.columns:
        raise DemandForecastError("历史订单数据必须包含 date 和 order_count 字段。")

    normalized = frame[["date", "order_count"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["order_count"] = pd.to_numeric(normalized["order_count"], errors="coerce")
    normalized = normalized.dropna().sort_values("date").drop_duplicates("date")
    if len(normalized) < 30:
        raise DemandForecastError("至少需要 30 条有效历史订单数据。")
    return normalized.reset_index(drop=True)


def _moving_average_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    history = [float(value) for value in values]
    window = min(7, len(history))
    result = []
    for _ in range(horizon):
        forecast = float(np.mean(history[-window:]))
        result.append(forecast)
        history.append(forecast)
    return np.asarray(result, dtype=float)


def _holt_winters_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
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


def _fit_forecast(model_name: str, values: np.ndarray, horizon: int) -> np.ndarray:
    if model_name == "moving_average":
        return _moving_average_forecast(values, horizon)
    if model_name == "holt_winters":
        return _holt_winters_forecast(values, horizon)
    if model_name == "arima":
        return _arima_forecast(values, horizon)
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
    frame = _normalize_history(history)
    horizon = int(horizon)
    if horizon < 1 or horizon > 120:
        raise DemandForecastError("预测期数必须在 1 到 120 之间。")

    model = (model or "auto").strip().lower()
    if model == "holt-winters":
        model = "holt_winters"
    candidate_models = ALL_MODELS if model == "auto" else (model,)
    if any(name not in ALL_MODELS for name in candidate_models):
        raise DemandForecastError(
            "model 只能是 auto、moving_average、holt_winters 或 arima。"
        )

    values = frame["order_count"].to_numpy(dtype=float)
    validation_horizon = min(max(horizon, 7), max(7, len(values) // 5))
    train_values = values[:-validation_horizon]
    actual_values = values[-validation_horizon:]
    if len(train_values) < 20:
        raise DemandForecastError("回测训练数据不足，请增加历史数据量。")

    candidates = {}
    for candidate in candidate_models:
        try:
            predicted = _fit_forecast(candidate, train_values, validation_horizon)
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
    future_values = _fit_forecast(selected_model, values, horizon)
    future_values = np.maximum(future_values, 0.0)

    last_date = frame["date"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

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
        "predicted_total": round(float(np.sum(future_values)), 2),
        "metrics": selected_metrics,
        "candidate_metrics": candidates,
        "history": history_payload,
        "forecast": forecast_payload,
    }
