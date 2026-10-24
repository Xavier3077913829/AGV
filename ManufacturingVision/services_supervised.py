# -*- coding: utf-8 -*-
"""视觉检测 —— 监督式 DeepCrack 分割（新增）

与 Frangi（传统、免训练、可解释）和 PatchCore（无监督、只需正常样本）互补：
DeepCrack 是**有监督**裂纹分割网络（VGG 编码器 + 多尺度侧输出融合），
在 DeepCrack 数据集（537 训练 / 237 测试）上训练后可给出像素级裂纹概率图。

权重查找顺序：
    models/deepcrack.pth  →  runs/deepcrack/deepcrack.pth  →  vendor/deepcrack/pretrained_net_G.pth

用法::

    from ManufacturingVision.services_supervised import segment_crack
    result = segment_crack("test.jpg")     # 无权重时抛 RuntimeError（含训练命令）
"""
from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "models", "deepcrack.pth"),
    os.path.join(PROJECT_ROOT, "runs", "deepcrack", "deepcrack.pth"),
    os.path.join(PROJECT_ROOT, "vendor", "deepcrack", "pretrained_net_G.pth"),
)

DEEPCRACK_DEFAULTS = {
    "algorithm": "deepcrack",
    "threshold": 0.72,
    "min_area": 8,
    "min_coverage": 0.0002,
    "max_side": 512,
    "device": "cpu",
}

_model = None
_weight_path = None
_device = "cpu"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def find_weights() -> str | None:
    for p in WEIGHT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def status() -> dict:
    w = find_weights()
    return {
        "torch": torch_available(),
        "weights": w,
        "weights_found": bool(w),
        "dataset": _dataset_stat(),
        "train_command": "python scripts/train_deepcrack.py --epochs 200",
    }


def _dataset_stat() -> dict:
    base = os.path.join(PROJECT_ROOT, "datasets", "vision", "deepcrack")
    out = {}
    for d in ("train_img", "train_lab", "test_img", "test_lab"):
        p = os.path.join(base, d)
        out[d] = len(os.listdir(p)) if os.path.isdir(p) else 0
    return out


def get_model(device: str = "cpu"):
    """懒加载 DeepCrack（有权重则加载）。"""
    global _model, _weight_path, _device
    if _model is not None and _device == device:
        return _model
    try:
        from .deepcrack_model import CrackSegmenter
    except Exception as exc:                       # noqa: BLE001
        raise RuntimeError(f"DeepCrack 模型不可用：{type(exc).__name__}: {exc}") from exc

    w = find_weights()
    if w is None:
        raise RuntimeError(
            "未找到 DeepCrack 权重。先训练：\n"
            "  python scripts/train_deepcrack.py --epochs 200\n"
            "或把官方 pretrained_net_G.pth 放到 models/deepcrack.pth"
        )
    _model = CrackSegmenter(weights_path=w, device=device)
    _weight_path = w
    _device = device
    return _model


def supervised_available() -> bool:
    """DeepCrack 是否可直接使用（torch 可用且存在权重）。"""
    return torch_available() and find_weights() is not None


def _merged_supervised(overrides) -> dict:
    """合并 DeepCrack 默认参数，只接受已知键并做范围校验。"""
    params = dict(DEEPCRACK_DEFAULTS)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key not in params:
                continue
            try:
                if isinstance(params[key], str):
                    params[key] = str(value).strip()
                elif isinstance(params[key], bool):
                    params[key] = str(value).strip().lower() not in {"0", "false", "no", ""}
                elif isinstance(params[key], int):
                    params[key] = int(round(float(value)))
                else:
                    params[key] = float(value)
            except (TypeError, ValueError):
                continue
    params["threshold"] = min(max(float(params["threshold"]), 0.05), 0.95)
    params["min_area"] = max(0, int(params["min_area"]))
    params["min_coverage"] = min(max(float(params["min_coverage"]), 0.0), 1.0)
    params["max_side"] = max(128, int(params["max_side"]))
    params["device"] = str(params["device"]).strip() or "cpu"
    return params


def analyze_crack_supervised(image, params=None) -> dict:
    """用 DeepCrack 对 BGR 图像做裂纹分割，返回与 analyze_crack 对齐的结果。"""
    import time

    import cv2
    import numpy as np

    started = time.perf_counter()
    resolved = _merged_supervised(params)
    device = str(resolved["device"])
    model = get_model(device=device)

    original_h, original_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if longest > int(resolved["max_side"]):
        scale = float(resolved["max_side"]) / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)

    prob = np.asarray(model.predict(rgb), dtype=np.float32)
    threshold = float(resolved["threshold"])
    binary = (prob >= threshold).astype(np.uint8)
    _total, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_area = int(resolved["min_area"])
    boxes = []
    for index in range(1, _total):
        x, y, box_w, box_h, area = stats[index]
        if area < min_area:
            continue
        boxes.append([int(x), int(y), int(box_w), int(box_h)])

    coverage = float(binary.mean())
    max_prob = float(prob.max()) if prob.size else 0.0
    mean_prob = float(prob.mean()) if prob.size else 0.0
    is_crack = coverage >= float(resolved["min_coverage"]) and bool(boxes)

    mask = (binary * 255).astype(np.uint8)
    if scale != 1.0:
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        boxes = [
            [
                int(round(x / scale)),
                int(round(y / scale)),
                int(round(box_w / scale)),
                int(round(box_h / scale)),
            ]
            for x, y, box_w, box_h in boxes
        ]
    if not is_crack:
        boxes = []
        mask[:] = 0

    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "task_type": "crack",
        "result": "crack" if is_crack else "normal",
        "count": len(boxes),
        "localized": bool(boxes),
        "boxes": boxes,
        "mask": mask,
        "score": round(max_prob, 4),
        "max_response": round(max_prob, 4),
        "strongest_component": round(coverage, 4),
        "coverage": round(coverage, 5),
        "max_prob": round(max_prob, 4),
        "mean_prob": round(mean_prob, 4),
        "algorithm": "deepcrack",
        "model": "DeepCrack (supervised)",
        "weights": _weight_path,
        "width": original_w,
        "height": original_h,
        "params": resolved,
        "elapsed_ms": round(elapsed, 2),
    }


def annotate_crack_supervised(image, analysis: dict):
    """在结果图上叠加裂纹掩膜、外接框与判定信息。"""
    import cv2

    canvas = image.copy()
    mask = analysis.get("mask")
    if mask is not None and mask.any():
        overlay = canvas.copy()
        overlay[mask > 0] = (0, 0, 255)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        for x, y, box_w, box_h in analysis.get("boxes", []):
            cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (0, 0, 255), 2)

    hit = analysis.get("result") == "crack"
    coverage = analysis.get("coverage", 0.0)
    text = f"DeepCrack {'crack' if hit else 'normal'} cov={coverage:.4f}"
    color = (0, 0, 255) if hit else (46, 160, 67)
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 310), 28), color, -1)
    cv2.putText(
        canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return canvas


def segment_crack(image, device: str = "cpu", threshold: float = 0.72) -> dict:
    """对单张图做裂纹分割，返回概率图统计 + 掩膜 + 判定。"""
    import numpy as np

    model = get_model(device=device)
    prob = model.predict(image)
    prob = np.asarray(prob, dtype=np.float32)
    mask = (prob >= threshold).astype(np.uint8) * 255
    coverage = float((prob >= threshold).mean())
    return {
        "model": "DeepCrack (supervised, VGG encoder + multi-scale side outputs)",
        "weights": _weight_path,
        "threshold": threshold,
        "coverage": round(coverage, 5),
        "max_prob": round(float(prob.max()), 4),
        "mean_prob": round(float(prob.mean()), 4),
        "result": "crack" if (coverage >= 0.002 or float(prob.max()) >= 0.90) else "normal",
        "mask_png_bytes": _encode_mask(mask),
        "shape": list(prob.shape),
    }


def _encode_mask(mask) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", mask)
    return buf.tobytes() if ok else b""
