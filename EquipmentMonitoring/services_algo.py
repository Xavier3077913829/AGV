# -*- coding: utf-8 -*-
"""设备状态监测 —— 算法增强层（新增）

在原 EquipmentMonitoring/services.py（StandardScaler + RandomForestClassifier + 阈值规则）
之外，新增两条可对比的算法线：

1) tsfresh  —— 从传感器时序窗口自动提取数百个统计特征（替代手写特征工程）
2) PyOD     —— 多种无监督异常检测器（IForest / LOF / ECOD / COPOD / HBOS ...）

设计要点
- pyod / tsfresh 都是**懒加载**：未安装时本模块仍可 import，项目不会崩；
- 只依赖 numpy / pandas / sklearn（项目已有），不引入 Django 依赖，便于单独测试；
- 标签列自动识别（Machine failure / machine_failure / failure / label ...）。

用法（脚本里）::

    from EquipmentMonitoring.services_algo import compare_detectors
    df = pd.read_csv("datasets/equipment/ai4i_equipment.csv")
    print(compare_detectors(df))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- 懒加载封装 -------------------------------------------------------------
try:
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF

    _PYOD_OK = True
except Exception:                                    # pragma: no cover
    _PYOD_OK = False

_TSFRESH_MSG = "未安装 tsfresh，请运行 scripts/install_algo.bat"

# 标签列候选（大小写/下划线都不敏感）
LABEL_CANDIDATES = ("machine failure", "machine_failure", "failure", "label",
                    "target", "is_failure", "anomaly")
# 明确要排除的非特征列
DROP_HINTS = ("udi", "product id", "product_id", "type", "timestamp", "time", "id")


def pyod_available() -> bool:
    return _PYOD_OK


def find_label_column(df: pd.DataFrame) -> str | None:
    """自动找出标签列；找不到返回 None。"""
    norm = {x.replace(" ", "").replace("_", "") for x in LABEL_CANDIDATES}
    for c in df.columns:
        if str(c).strip().lower().replace(" ", "").replace("_", "") in norm:
            return c
    return None


def pick_sensor_columns(df: pd.DataFrame, label_col: str | None = None) -> list[str]:
    """挑出数值型传感器列，排除标签列 / ID 列 / 类型列 / 故障子标签。"""
    label = label_col or find_label_column(df)
    drop = {str(x).strip().lower() for x in DROP_HINTS}
    if label:
        drop.add(str(label).strip().lower())
    cols = []
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in drop or "failure" in lc:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_detectors(contamination: float = 0.034) -> dict:
    """返回 {名字: 检测器实例}。默认污染率 ≈ AI4I 数据集故障率。"""
    if not _PYOD_OK:
        raise RuntimeError("pyod 未安装：请先 pip install ./vendor/pyod")
    return {
        "IForest": IForest(contamination=contamination, n_estimators=200, random_state=42),
        "ECOD": ECOD(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "LOF": LOF(contamination=contamination, n_neighbors=20),
    }


def detect_anomalies(df: pd.DataFrame, method: str = "IForest", contamination: float = 0.034):
    """无监督异常检测，返回 (score, label, model)。"""
    from sklearn.preprocessing import StandardScaler

    cols = pick_sensor_columns(df)
    X = StandardScaler().fit_transform(df[cols].to_numpy(dtype=float))
    det = build_detectors(contamination)[method]
    det.fit(X)
    return det.decision_scores_, det.labels_, det


def round_series_to_windows(df: pd.DataFrame, window: int = 64, step: int | None = None):
    """把逐行快照数据切成滑窗（id/time/value 长表），供 tsfresh 提特征。"""
    step = step or window
    cols = pick_sensor_columns(df)
    frames = []
    for start in range(0, len(df) - window + 1, step):
        seg = df.iloc[start:start + window][cols].reset_index(drop=True)
        frames.append(pd.DataFrame({
            "id": np.repeat(np.arange(len(cols)), window),
            "time": np.tile(np.arange(window), len(cols)),
            "value": seg.to_numpy(dtype=float).T.ravel(),
            "kind": np.repeat(np.array(cols, dtype=object), window),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_tsfresh_features(df: pd.DataFrame, window: int = 64):
    """用 tsfresh 自动提特征，返回 DataFrame（行=窗口，列=特征）。"""
    try:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
    except Exception as e:                            # pragma: no cover
        raise RuntimeError(_TSFRESH_MSG) from e

    long = round_series_to_windows(df, window=window)
    if long.empty:
        raise RuntimeError(f"数据行数不足以切出窗口（window={window}）")
    feats = extract_features(
        long, column_id="id", column_sort="time", column_kind="kind",
        column_value="value", n_jobs=0, disable_progressbar=True,
    )
    return impute(feats)


def binarize_labels(series) -> np.ndarray:
    """把标签转成 0/1（异常=1）。

    - 数值型：直接取整（0/1）；
    - 字符串型（如本项目 label = normal / fault / worn）：非 normal 记为 1。
    """
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    txt = series.astype(str).str.strip().str.lower()
    return (txt != "normal").astype(int).to_numpy()


def compare_detectors(df: pd.DataFrame, label_col: str | None = None):
    """同口径对比：RF 基线 vs 各 PyOD 检测器，返回 DataFrame（可直接进报告）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    label = label_col or find_label_column(df)
    if not label:
        raise RuntimeError(f"未找到标签列，候选：{LABEL_CANDIDATES}")

    cols = pick_sensor_columns(df, label)
    X = df[cols].to_numpy(dtype=float)
    y = binarize_labels(df[label])
    if y.sum() == 0:
        raise RuntimeError(f"标签列 {label} 全为正常样本，无法做异常检测对比")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rows = []
    # 基线：有监督 RF（项目现有做法）
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)
    p = rf.predict_proba(sc.transform(Xte))[:, 1]
    rows.append({"模型": "RandomForest(基线, 有监督)",
                 "ROC-AUC": roc_auc_score(yte, p),
                 "F1": f1_score(yte, (p >= 0.5).astype(int))})

    # PyOD：无监督（只在测试集拟合，模拟"无标签"场景）
    if not _PYOD_OK:
        print("[提示] pyod 未安装，仅输出 RF 基线。安装后重跑可得到 5 个无监督检测器结果。")
        out = pd.DataFrame(rows)
        out["ROC-AUC"] = out["ROC-AUC"].round(4)
        out["F1"] = out["F1"].round(4)
        return out

    cont = float(y.mean())
    cand = {"IForest": IForest, "ECOD": ECOD, "HBOS": HBOS, "LOF": LOF, "COPOD": COPOD}
    for name, cls in cand.items():
        try:
            try:
                det = cls(contamination=cont, random_state=42) if name == "IForest" else cls(contamination=cont)
            except TypeError:
                det = cls(contamination=cont)
            sc2 = StandardScaler().fit(Xte)
            det.fit(sc2.transform(Xte))
            rows.append({"模型": f"PyOD-{name}(无监督)",
                         "ROC-AUC": roc_auc_score(yte, det.decision_scores_),
                         "F1": f1_score(yte, det.labels_)})
        except Exception as e:                       # pragma: no cover
            rows.append({"模型": f"PyOD-{name}", "ROC-AUC": float("nan"), "F1": float("nan")})
            print(f"[warn] {name} 失败: {e}")

    out = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False, ignore_index=True)
    out["ROC-AUC"] = out["ROC-AUC"].round(4)
    out["F1"] = out["F1"].round(4)
    return out


# ======================= 业务集成（供 views_algo 调用） =======================

def tsfresh_available() -> bool:
    try:
        import tsfresh  # noqa: F401
        return True
    except Exception:
        return False


def features_summary(df: pd.DataFrame, window: int = 64, top_k: int = 12) -> dict:
    """tsfresh 自动特征摘要（窗口数 / 特征数 / 样例列 / 矩阵）。"""
    feats = extract_tsfresh_features(df, window=window)
    return {
        "windows": int(len(feats)),
        "feature_count": int(feats.shape[1]),
        "sample_columns": [str(c) for c in list(feats.columns)[:top_k]],
        "feature_matrix": feats.round(4).to_dict(orient="records"),
    }


def diagnose_with_algo(records, equipment_code: str = "CNC-01",
                       data_source: str = "ai4i_equipment",
                       method: str = "ECOD") -> dict:
    """原 diagnose_equipment（RF + 规则）之上叠加 PyOD 无监督异常分数。

    返回原结果 + ``algorithm_layer`` + ``anomaly`` 两个新字段；
    pyod 未安装时只返回原结果（不报错），保证线上可用。
    """
    from .services import diagnose_equipment

    base = diagnose_equipment(records, equipment_code, data_source)
    merged = dict(base)
    merged["algorithm_layer"] = {
        "pyod_available": pyod_available(),
        "tsfresh_available": tsfresh_available(),
        "method": method if pyod_available() else None,
    }
    if not pyod_available():
        merged["anomaly"] = {"error": "pyod 未安装，请 pip install ./vendor/pyod"}
        return merged

    frame = pd.DataFrame(list(records)) if not isinstance(records, pd.DataFrame) else records
    try:
        scores, labels, _ = detect_anomalies(frame, method=method)
        merged["anomaly"] = {
            "method": method,
            "scores": [round(float(s), 4) for s in scores],
            "labels": [int(v) for v in labels],
            "flagged": int(sum(labels)),
        }
    except Exception as exc:                     # pragma: no cover
        merged["anomaly"] = {"error": f"{type(exc).__name__}: {exc}"}
    return merged
