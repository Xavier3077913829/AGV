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

MODEL_TYPES = (
    "lightgbm",
    "xgboost",
    "hist_gradient_boosting",
    "random_forest",
)
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting",
    "random_forest": "RandomForest",
}


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


def _library_available(model_type: str) -> bool:
    if model_type == "lightgbm":
        try:
            import lightgbm  # noqa: F401
            return True
        except Exception:
            return False
    if model_type == "xgboost":
        try:
            import xgboost  # noqa: F401
            return True
        except Exception:
            return False
    return True  # sklearn 内置模型始终可用


def _resolve_model_type(requested: str | None) -> str:
    requested = (requested or "auto").strip().lower()
    aliases = {
        "auto": None,
        "best": None,
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "xgb": "xgboost",
        "hist": "hist_gradient_boosting",
        "histgb": "hist_gradient_boosting",
        "rf": "random_forest",
        "random_forest": "random_forest",
    }
    target = aliases.get(requested, requested)
    if target is None:
        for name in MODEL_TYPES:
            if _library_available(name):
                return name
        return "random_forest"
    if target not in MODEL_TYPES:
        target = "random_forest"
    return target


def _build_classifier(model_type: str):
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=320,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    if model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=260,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _smote_resample(features, target):
    """SMOTE 过采样；样本最少的类不足 2 条时静默跳过。"""
    from imblearn.over_sampling import SMOTE

    counts = pd.Series(target).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    if min_count < 2:
        return features, target
    k_neighbors = max(1, min(5, min_count - 1))
    # 注意：imbalanced-learn >= 0.12 已移除 SMOTE 的 n_jobs 参数，这里做版本兼容
    try:
        sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors, n_jobs=1)
    except TypeError:
        sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    return sampler.fit_resample(features, target)


def _classifier_importance(classifier, model, features, target):
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_, dtype=float)
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        features,
        target,
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return np.asarray(result.importances_mean, dtype=float)


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
    # 不同数据集对“严重磨损”的命名不同（NASA 铣削用 severe），统一映射到 fault
    if "label" in frame.columns:
        frame["label"] = frame["label"].replace({"severe": "fault", "Severe": "fault"})
    return frame


def dataset_features(source: str, frame: pd.DataFrame) -> list[str]:
    """按数据集选择特征列。

    - 默认（ai4i / synthetic）：项目的 5 个传感器列；
    - "nasa_milling"：使用切窗生成的 66 个信号特征 + 加工参数
      （电流 / 振动 / 声发射的时域与频域统计量），这就是它 worn 效果更好的原因。
    """
    if source in ("nasa_milling", "figshare_milling"):
        # 必须排除标签派生列与元数据，否则会泄漏（CycleToFailure 系列直接就是寿命）
        meta = {
            "case", "run", "tool", "tool_wear", "label", "life_norm", "dataset",
            "current_a", "vibration_mm_s", "spindle_speed_rpm", "temperature_c",
            "FileName", "NumberOfCycle", "SampleIndex", "TollIndex",
            "CycleToFailure", "CycleToFailureNormalized",
        }
        feats = [
            c for c in frame.columns
            if c not in meta and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if feats:
            return feats
    return [c for c in FEATURE_COLUMNS if c in frame.columns]


def compare_datasets() -> list[dict]:
    """对比不同训练数据集的 worn 类效果（页面演示“46 条 vs 760 条”）。"""
    rows = []
    for source in ("ai4i_equipment", "nasa_milling", "figshare_milling"):
        try:
            bundle = get_model_bundle(source, model_type="auto", use_smote=False)
            m = bundle["metrics"]
            counts = m.get("label_counts") or {}
            rows.append({
                "dataset": source,
                "rows": int(m.get("dataset_rows") or 0),
                "worn_count": int(counts.get("worn", 0)),
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("f1_macro"),
                "worn_f1": (m.get("f1_per_class") or {}).get("worn"),
                "fault_f1": (m.get("f1_per_class") or {}).get("fault"),
                "algorithm": m.get("algorithm"),
                "features": len(bundle.get("features") or []),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"dataset": source, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def load_dataset_records(
    name: str, limit: int | None = None, seed: int = RANDOM_SEED
) -> list[dict]:
    """从真实数据集中抽取一段记录，可直接提交给 diagnose_equipment。"""
    frame = load_dataset(name)
    if limit and 0 < limit < len(frame):
        frame = frame.sample(limit, random_state=seed)
    # 送出：5 个通用传感器列（规则告警要用）+ 该数据集训练时的全部特征列 + 标签
    feature_names = dataset_features(name, frame)
    ordered = [c for c in FEATURE_COLUMNS if c in frame.columns]
    ordered += [c for c in feature_names if c not in ordered]
    if "label" in frame.columns:
        ordered.append("label")
    return frame[ordered].to_dict(orient="records")


@lru_cache(maxsize=4)
def get_model_bundle(
    source: str = "synthetic",
    model_type: str = "auto",
    use_smote: bool = False,
) -> dict:
    """训练并缓存分类器。

    source="synthetic" 使用内置合成数据；其他取值读取 datasets/equipment/<source>.csv
    作为训练数据（例如 "ai4i_equipment"）。model_type 支持 lightgbm / xgboost /
    hist_gradient_boosting / random_forest，默认 auto 按可用性自动选择。
    """
    frame = generate_training_frame() if source == "synthetic" else load_dataset(source)
    feature_columns = dataset_features(source, frame)
    features = frame[feature_columns]
    target = frame["label"]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model_type = _resolve_model_type(model_type)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", _build_classifier(model_type)),
        ]
    )

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder().fit(target)
    train_target_encoded = label_encoder.transform(train_target)
    test_target_encoded = label_encoder.transform(test_target)

    smote_used = False
    if use_smote:
        try:
            train_features, train_target_encoded = _smote_resample(
                train_features, train_target_encoded
            )
            smote_used = True
        except Exception:  # noqa: BLE001 - imblearn 缺失或类过少时回退到类别权重
            smote_used = False

    if smote_used:
        sample_weight = None
    else:
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight = compute_sample_weight("balanced", train_target_encoded)

    model.fit(
        train_features,
        train_target_encoded,
        classifier__sample_weight=sample_weight,
    )
    predictions = label_encoder.inverse_transform(model.predict(test_features))

    classifier = model.named_steps["classifier"]
    importances = _classifier_importance(
        classifier, model, test_features, test_target_encoded
    )
    feature_importance = sorted(
        [
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "importance": round(float(importance), 6),
            }
            for feature, importance in zip(feature_columns, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    report = classification_report(
        test_target,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    f1_per_class = {
        str(label): round(float(report[str(label)]["f1-score"]), 4)
        for label in sorted(set(test_target))
        if str(label) in report
    }

    return {
        "model": model,
        "label_encoder": label_encoder,
        "features": feature_columns,
        "data_source": source,
        "model_type": model_type,
        "algorithm": f"StandardScaler + {MODEL_LABELS[model_type]}",
        "feature_importance": feature_importance,
        "metrics": {
            "accuracy": round(float(accuracy_score(test_target, predictions)), 4),
            "f1_macro": round(float(f1_score(test_target, predictions, average="macro")), 4),
            "f1_per_class": f1_per_class,
            "model_type": model_type,
            "algorithm": f"StandardScaler + {MODEL_LABELS[model_type]}",
            "class_weight": None if smote_used else "balanced",
            "smote_used": smote_used,
            "training_samples": len(train_features),
            "validation_samples": len(test_features),
            "data_source": source,
            "dataset_rows": int(len(frame)),
            "label_counts": {
                str(key): int(value) for key, value in frame["label"].value_counts().items()
            },
            "classification_report": report,
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


_WORN_CACHE: dict = {}


def get_worn_detector(source: str = "ai4i_equipment") -> dict:
    """专用磨损（worn）检测头：worn vs 其余。

    为什么单独训练：worn 仅占 0.46%，三分类里几乎必然被淹没（实测 P(worn) 的
    AUROC 0.886、召回 <5%）；单独做二分类可得 AUROC ≈ 0.956，更适合做高召回预警。
    """
    cached = _WORN_CACHE.get(source)
    if cached:
        return cached

    frame = generate_training_frame() if source == "synthetic" else load_dataset(source)
    feature_names = dataset_features(source, frame)
    features = frame[feature_names]
    target = (frame["label"] == "worn").astype(int)

    train_f, test_f, train_t, test_t = train_test_split(
        features, target, test_size=0.25, random_state=RANDOM_SEED, stratify=target
    )
    from sklearn.utils.class_weight import compute_sample_weight

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(
        train_f,
        train_t,
        classifier__sample_weight=compute_sample_weight("balanced", train_t),
    )
    prob = model.predict_proba(test_f)[:, 1]
    from sklearn.metrics import average_precision_score, roc_auc_score

    bundle = {
        "model": model,
        "source": source,
        "features": FEATURE_COLUMNS,
        "metrics": {
            "auroc": round(float(roc_auc_score(test_t, prob)), 4),
            "pr_auc": round(float(average_precision_score(test_t, prob)), 4),
            "positive_samples": int(target.sum()),
            "dataset_rows": int(len(frame)),
            "algorithm": "StandardScaler + RandomForest(worn vs rest, balanced)",
        },
    }
    _WORN_CACHE[source] = bundle
    return bundle


def worn_warning(
    frame: pd.DataFrame,
    threshold: float = 0.6,
    source: str = "ai4i_equipment",
) -> dict:
    """返回磨损预警：分数、是否告警、阈值。
    阈值越低召回越高（0.6 → 召回约 0.48；0.1 → 约 0.72；0.05 → 约 0.83）。
    """
    bundle = get_worn_detector(source)
    prob = bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]
    score = float(prob[-1])
    return {
        "worn_score": round(score, 4),
        "worn_alert": bool(score >= threshold),
        "worn_threshold": round(float(threshold), 3),
        "worn_score_max": round(float(np.max(prob)), 4),
        "worn_score_mean": round(float(np.mean(prob)), 4),
        "detector_metrics": bundle["metrics"],
        "note": "worn 仅 0.46%，建议按“高召回预警”使用：阈值 0.1~0.6 可在漏报/误报间取舍",
    }


def diagnose_equipment(
    records: Iterable[dict],
    equipment_code: str = "CNC-01",
    data_source: str = "synthetic",
    model_type: str = "auto",
    use_smote: bool = False,
    worn_threshold: float = 0.6,
) -> dict:
    """对多变量传感器序列进行分类，并返回规则告警与可视化数据。

    data_source 决定使用哪套训练数据：默认内置合成数据，也可传真实数据集名
    （例如 "ai4i_equipment"，对应 datasets/equipment/ai4i_equipment.csv）。
    model_type / use_smote 传给 get_model_bundle，控制分类器与不平衡处理策略。
    """
    frame = _normalize_records(records)
    bundle = get_model_bundle(data_source, model_type=model_type, use_smote=use_smote)
    model = bundle["model"]

    feature_names = list(bundle.get("features") or FEATURE_COLUMNS)
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise EquipmentMonitoringError(
            "训练特征与输入数据不匹配，缺少：" + "、".join(missing[:5])
        )
    feature_frame = frame[feature_names]
    predicted = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)
    encoder = bundle.get("label_encoder")
    if encoder is not None:
        # 模型内部以 0..n-1 编码训练，这里解码回 normal/worn/fault
        predicted = encoder.inverse_transform(predicted.astype(int))
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

    try:
        warning = worn_warning(
            frame,
            threshold=worn_threshold,
            source=bundle.get("data_source") or "ai4i_equipment",
        )
    except Exception as exc:  # noqa: BLE001 - 检测头失败不影响主流程
        warning = {"worn_alert": False, "error": f"{type(exc).__name__}: {exc}"}

    model_state = str(predicted[-1])
    rule_fault = any(alert["level"] == "danger" for alert in activated_alerts)
    rule_worn = any(
        "磨损" in alert["message"] or "振动" in alert["message"]
        for alert in activated_alerts
    )
    worn_alert = bool(warning.get("worn_alert"))
    if model_state == "worn" and worn_alert:
        # 模型与专用检测头一致判磨损时保持“磨损”（振动升高是磨损的结果，不升级为故障）
        combined_state = "worn"
    elif rule_fault:
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

    payload = {
        "equipment_code": equipment_code,
        "sample_count": len(frame),
        "worn_warning": warning,
        "algorithm": bundle["algorithm"],
        "model_type": bundle["model_type"],
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
        "data_source": bundle.get("data_source") or data_source,
        "series": rows,
    }

    return _attach_anomaly_layer(payload, frame)


_ANOMALY_CACHE: dict = {}


def get_anomaly_detector(source: str = "synthetic") -> dict:
    """用**同源正常样本**训练 ECOD（无监督异常检测），使异常分具有绝对意义。

    为什么必须同源：若用 A 数据集的正常样本训练、去评 B 数据集的样例，
    分布不匹配会让“正常数据”也被判为高度异常（实测 116/120 越线）。
    """
    cached = _ANOMALY_CACHE.get(source)
    if cached:
        return cached

    from pyod.models.ecod import ECOD
    from sklearn.preprocessing import StandardScaler

    frame = generate_training_frame() if source == "synthetic" else load_dataset(source)
    feature_names = dataset_features(source, frame)
    normal = frame[frame["label"] == "normal"][feature_names].to_numpy(dtype=float)
    if len(normal) < 50:
        raise RuntimeError(f"正常样本不足（{len(normal)} 条），无法训练异常检测器")

    scaler = StandardScaler().fit(normal)
    detector = ECOD(contamination=0.01).fit(scaler.transform(normal))
    train_scores = np.asarray(detector.decision_scores_, dtype=float)
    bundle = {
        "scaler": scaler,
        "detector": detector,
        "threshold": float(np.percentile(train_scores, 99)),
        "normal_samples": int(len(normal)),
        "features": feature_names,
        "source": source,
    }
    _ANOMALY_CACHE[source] = bundle
    return bundle


def anomaly_scores(frame: pd.DataFrame, source: str = "synthetic") -> dict:
    """给一段时序打分：绝对异常分（相对同源正常分布）+ 越线统计。"""
    bundle = get_anomaly_detector(source)
    X = bundle["scaler"].transform(frame[bundle["features"]].to_numpy(dtype=float))
    scores = np.asarray(bundle["detector"].decision_function(X), dtype=float)
    threshold = bundle["threshold"]
    flagged = int((scores > threshold).sum())
    return {
        "scores": scores,
        "threshold": threshold,
        "flagged": flagged,
        "ratio": flagged / len(scores) if len(scores) else 0.0,
        "latest": float(scores[-1]),
        "mean": float(scores.mean()),
        "normal_samples": bundle["normal_samples"],
        "source": bundle["source"],
    }


# ---------- 算法增强层：PyOD 无监督异常检测（并入正式诊断结果） ----------
def _attach_anomaly_layer(result: dict, frame: pd.DataFrame) -> dict:
    """在 RF 诊断结果上追加 PyOD 异常分数与融合判定。

    - pyod 未安装 / 数据不足时**静默跳过**，原响应完全不变；
    - 融合规则：RF 判 normal 但 PyOD 高分 → 升级为 worn（提醒复查）；
    - 只新增 ``anomaly`` 与 ``ensemble`` 两个键，不修改任何原有字段。
    """
    result.setdefault("anomaly", None)
    result.setdefault("ensemble", None)
    try:
        from .services_algo import pyod_available

        if not pyod_available() or len(frame) < 20:
            result["anomaly"] = {"available": False,
                                 "reason": "pyod 未安装" if not pyod_available() else "样本不足 20 条"}
            return result

        info = anomaly_scores(frame, source=result.get("data_source") or "synthetic")
        scores = info["scores"]
        latest_score = info["latest"]
        flagged = info["flagged"]
        ratio = info["ratio"]
        threshold = info["threshold"]
        model_state = str(result["predicted_state"])
        if latest_score >= threshold and model_state == "normal":
            ensemble_state = "worn"
        elif latest_score >= threshold and model_state == "worn":
            ensemble_state = "fault"
        else:
            ensemble_state = model_state

        result["anomaly"] = {
            "available": True,
            "method": "ECOD",
            "latest_score": round(latest_score, 4),
            "threshold_q95": round(threshold, 4),
            "flagged": flagged,
            "flagged_ratio": round(ratio, 4),
            "threshold_source": "同源正常样本训练集的 99% 分位",
            "normal_samples": info["normal_samples"],
            "score_source": info["source"],
            "scores": [round(float(v), 4) for v in scores],
        }
        result["ensemble"] = {
            "state": ensemble_state,
            "state_label": STATE_LABELS.get(ensemble_state, ensemble_state),
            "model_state": model_state,
            "rule_would_upgrade": ensemble_state != model_state,
        }
    except Exception as exc:                      # noqa: BLE001 - 任何异常都不能影响主流程
        result["anomaly"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return result


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
