from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "current_a",
    "spindle_speed_rpm",
    "tool_wear",
]

FEATURE_LABELS = {
    "temperature_c": "温度",
    "vibration_mm_s": "振动",
    "current_a": "电流",
    "spindle_speed_rpm": "主轴转速",
    "tool_wear": "刀具磨损",
}

STATE_LABELS = {
    "normal": "正常",
    "worn": "磨损",
    "fault": "故障",
}

RANDOM_SEED = 42


class EquipmentMonitoringError(ValueError):
    """设备监测输入或计算异常。"""


def _class_rows(
    label: str,
    count: int,
    means: dict[str, float],
    standard_deviations: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    data = {}
    for feature in FEATURE_COLUMNS:
        values = rng.normal(
            means[feature],
            standard_deviations[feature],
            size=count,
        )
        lower_bound = 0.0
        data[feature] = np.clip(values, lower_bound, None)
    data["label"] = label
    return pd.DataFrame(data)


def generate_training_frame(per_class: int = 320) -> pd.DataFrame:
    """生成具有明确物理含义的三类设备状态训练数据。"""
    rng = np.random.default_rng(RANDOM_SEED)
    frames = [
        _class_rows(
            "normal",
            per_class,
            {
                "temperature_c": 62.0,
                "vibration_mm_s": 1.0,
                "current_a": 4.6,
                "spindle_speed_rpm": 1500.0,
                "tool_wear": 0.14,
            },
            {
                "temperature_c": 2.4,
                "vibration_mm_s": 0.28,
                "current_a": 0.28,
                "spindle_speed_rpm": 34.0,
                "tool_wear": 0.06,
            },
            rng,
        ),
        _class_rows(
            "worn",
            per_class,
            {
                "temperature_c": 68.0,
                "vibration_mm_s": 2.0,
                "current_a": 5.35,
                "spindle_speed_rpm": 1465.0,
                "tool_wear": 0.56,
            },
            {
                "temperature_c": 3.0,
                "vibration_mm_s": 0.48,
                "current_a": 0.42,
                "spindle_speed_rpm": 46.0,
                "tool_wear": 0.11,
            },
            rng,
        ),
        _class_rows(
            "fault",
            per_class,
            {
                "temperature_c": 76.0,
                "vibration_mm_s": 3.7,
                "current_a": 6.6,
                "spindle_speed_rpm": 1395.0,
                "tool_wear": 0.84,
            },
            {
                "temperature_c": 4.0,
                "vibration_mm_s": 0.8,
                "current_a": 0.62,
                "spindle_speed_rpm": 62.0,
                "tool_wear": 0.12,
            },
            rng,
        ),
    ]
    result = pd.concat(frames, ignore_index=True)
    return result.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "equipment"

# 真实数据集出处（报告与前端展示用）
DATASET_SOURCES = {
    "ai4i_equipment": "UCI AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0, DOI 10.24432/C5HS5C)",
    "phm2012_equipment": "FEMTO-ST / IEEE PHM 2012 Data Challenge 轴承全寿命数据",
}


def available_datasets() -> list[dict]:
    """列出 datasets/equipment 下可用于训练/演示的真实数据集。"""
    if not DATASET_DIR.is_dir():
        return []
    datasets = []
    for path in sorted(DATASET_DIR.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - 坏文件直接跳过，不影响其他数据集
            continue
        label_counts = (
            frame["label"].value_counts().to_dict() if "label" in frame.columns else {}
        )
        datasets.append(
            {
                "name": path.stem,
                "file": path.name,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "label_counts": {str(key): int(value) for key, value in label_counts.items()},
                "source": DATASET_SOURCES.get(path.stem, "本地数据集"),
            }
        )
    return datasets


def load_dataset(name: str) -> pd.DataFrame:
    """按名称读取真实数据集（自动忽略 .csv 后缀）。"""
    stem = name.strip()
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    path = DATASET_DIR / f"{stem}.csv"
    if not path.is_file():
        raise EquipmentMonitoringError(f"数据集不存在：{name}")
    frame = pd.read_csv(path)
    missing = [
        column
        for column in (*FEATURE_COLUMNS, "label")
        if column not in frame.columns
    ]
    if missing:
        raise EquipmentMonitoringError("数据集缺少字段：" + "、".join(missing))
    return frame


def load_dataset_records(
    name: str, limit: int | None = None, seed: int = RANDOM_SEED
) -> list[dict]:
    """从真实数据集中抽取一段记录，可直接提交给 diagnose_equipment。"""
    frame = load_dataset(name)
    if limit and 0 < limit < len(frame):
        frame = frame.sample(limit, random_state=seed)
    columns = [column for column in (*FEATURE_COLUMNS, "label") if column in frame.columns]
    return frame[columns].to_dict(orient="records")


@lru_cache(maxsize=4)
def get_model_bundle(source: str = "synthetic") -> dict:
    """训练并缓存随机森林分类器。

    source="synthetic" 使用内置合成数据；其他取值读取 datasets/equipment/<source>.csv
    作为训练数据（例如 "ai4i_equipment"）。
    """
    frame = generate_training_frame() if source == "synthetic" else load_dataset(source)
    features = frame[FEATURE_COLUMNS]
    target = frame["label"]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    classifier = model.named_steps["classifier"]
    feature_importance = sorted(
        [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "importance": round(float(importance), 6),
            }
            for feature, importance in zip(
                FEATURE_COLUMNS, classifier.feature_importances_
            )
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {
        "model": model,
        "features": FEATURE_COLUMNS,
        "data_source": source,
        "feature_importance": feature_importance,
        "metrics": {
            "accuracy": round(float(accuracy_score(test_target, predictions)), 4),
            "f1_macro": round(float(f1_score(test_target, predictions, average="macro")), 4),
            "training_samples": len(train_features),
            "validation_samples": len(test_features),
            "data_source": source,
            "dataset_rows": int(len(frame)),
            "label_counts": {
                str(key): int(value) for key, value in frame["label"].value_counts().items()
            },
            "classification_report": classification_report(
                test_target,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
    }


def _normalize_records(records: Iterable[dict]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))

    if frame.empty:
        raise EquipmentMonitoringError("传感器数据不能为空。")

    missing = [feature for feature in FEATURE_COLUMNS if feature not in frame.columns]
    if missing:
        raise EquipmentMonitoringError(
            "传感器数据缺少字段：" + "、".join(missing)
        )

    normalized = frame.copy()
    for feature in FEATURE_COLUMNS:
        normalized[feature] = pd.to_numeric(normalized[feature], errors="coerce")
    if normalized[FEATURE_COLUMNS].isna().any().any():
        raise EquipmentMonitoringError("传感器字段必须全部为数值。")
    return normalized


def _rule_alerts(row: pd.Series) -> list[dict]:
    alerts = []
    if row["temperature_c"] >= 74:
        alerts.append({"level": "warning", "message": "温度超过警戒值 74 ℃"})
    if row["vibration_mm_s"] >= 3.1:
        alerts.append({"level": "danger", "message": "振动超过警戒值 3.1 mm/s"})
    if row["current_a"] >= 6.2:
        alerts.append({"level": "warning", "message": "电流超过警戒值 6.2 A"})
    if row["tool_wear"] >= 0.78:
        alerts.append({"level": "danger", "message": "刀具磨损率超过 0.78"})
    if row["spindle_speed_rpm"] <= 1415:
        alerts.append({"level": "warning", "message": "主轴转速明显下降"})
    return alerts


def diagnose_equipment(
    records: Iterable[dict],
    equipment_code: str = "CNC-01",
    data_source: str = "synthetic",
) -> dict:
    """对多变量传感器序列进行分类，并返回规则告警与可视化数据。

    data_source 决定使用哪套训练数据：默认内置合成数据，也可传真实数据集名
    （例如 "ai4i_equipment"，对应 datasets/equipment/ai4i_equipment.csv）。
    """
    frame = _normalize_records(records)
    bundle = get_model_bundle(data_source)
    model = bundle["model"]

    feature_frame = frame[FEATURE_COLUMNS]
    predicted = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)
    classes = list(model.named_steps["classifier"].classes_)
    probability_by_class = {
        label: probabilities[:, index]
        for index, label in enumerate(classes)
    }

    rows = []
    activated_alerts = []
    for index, (_, row) in enumerate(frame.iterrows()):
        state = str(predicted[index])
        confidence = float(
            probability_by_class.get(state, np.zeros(len(frame)))[index]
        )
        alerts = _rule_alerts(row)
        activated_alerts.extend(alerts)
        rows.append(
            {
                "index": index,
                "timestamp": str(
                    row.get(
                        "timestamp",
                        datetime(2026, 1, 1, tzinfo=timezone.utc)
                        + timedelta(minutes=index),
                    )
                ),
                **{
                    feature: round(float(row[feature]), 4)
                    for feature in FEATURE_COLUMNS
                },
                "predicted_state": state,
                "predicted_state_label": STATE_LABELS[state],
                "confidence": round(confidence, 4),
                "rule_alerts": alerts,
            }
        )

    state_counts = Counter(str(item) for item in predicted)
    state_distribution = {
        label: {
            "count": int(state_counts.get(label, 0)),
            "ratio": round(state_counts.get(label, 0) / len(predicted), 4),
            "label": STATE_LABELS[label],
        }
        for label in ("normal", "worn", "fault")
    }

    model_state = str(predicted[-1])
    rule_fault = any(alert["level"] == "danger" for alert in activated_alerts)
    rule_worn = any(
        "磨损" in alert["message"] or "振动" in alert["message"]
        for alert in activated_alerts
    )
    if rule_fault:
        combined_state = "fault"
    elif rule_worn and model_state == "normal":
        combined_state = "worn"
    else:
        combined_state = model_state

    feature_summary = {
        feature: {
            "label": FEATURE_LABELS[feature],
            "mean": round(float(frame[feature].mean()), 4),
            "min": round(float(frame[feature].min()), 4),
            "max": round(float(frame[feature].max()), 4),
            "latest": round(float(frame[feature].iloc[-1]), 4),
        }
        for feature in FEATURE_COLUMNS
    }

    return {
        "equipment_code": equipment_code,
        "sample_count": len(frame),
        "predicted_state": combined_state,
        "predicted_state_label": STATE_LABELS[combined_state],
        "confidence": round(
            float(np.max(probabilities[-1])),
            4,
        ),
        "state_distribution": state_distribution,
        "rule_alerts": activated_alerts[-12:],
        "feature_summary": feature_summary,
        "feature_importance": bundle["feature_importance"],
        "model_metrics": bundle["metrics"],
        "series": rows,
    }


def generate_demo_series(
    scenario: str = "worn",
    points: int = 120,
    seed: int = 2026,
) -> list[dict]:
    """生成一段用于演示的设备传感器时间序列。"""
    scenario = scenario if scenario in STATE_LABELS else "worn"
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)

    if scenario == "normal":
        base = np.array([62.0, 1.0, 4.6, 1500.0, 0.14])
        growth = np.array([0.0, 0.0, 0.0, 0.0, 0.001])
        noise_scale = np.array([1.4, 0.15, 0.18, 22.0, 0.025])
    elif scenario == "fault":
        base = np.array([70.0, 2.2, 5.6, 1460.0, 0.42])
        growth = np.array([0.055, 0.018, 0.012, -0.55, 0.0045])
        noise_scale = np.array([2.0, 0.35, 0.28, 28.0, 0.035])
    else:
        base = np.array([63.0, 1.2, 4.8, 1492.0, 0.20])
        growth = np.array([0.04, 0.012, 0.008, -0.30, 0.0032])
        noise_scale = np.array([1.8, 0.26, 0.24, 24.0, 0.03])

    values = (
        base
        + np.arange(points)[:, None] * growth
        + rng.normal(0.0, 1.0, size=(points, len(FEATURE_COLUMNS)))
        * noise_scale
    )
    values[:, 4] = np.clip(values[:, 4], 0.0, 1.0)
    values[:, 0] = np.clip(values[:, 0], 20.0, 110.0)
    values[:, 1] = np.clip(values[:, 1], 0.0, 8.0)
    values[:, 2] = np.clip(values[:, 2], 0.0, 12.0)
    values[:, 3] = np.clip(values[:, 3], 600.0, 1800.0)

    return [
        {
            "timestamp": (start_time + timedelta(minutes=index)).isoformat(),
            **{
                feature: round(float(values[index, column]), 4)
                for column, feature in enumerate(FEATURE_COLUMNS)
            },
        }
        for index in range(points)
    ]
