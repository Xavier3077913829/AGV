# -*- coding: utf-8 -*-
"""制造视觉 —— 算法增强层（新增）：anomalib 无监督缺陷检测

现有 ManufacturingVision/services.py 用传统算法（多尺度 Frangi 黑脊 + 连通域筛选），
本模块在其之上提供 industrial 级的**无监督异常检测**能力：

- PatchCore（Anomalib 默认强基线）：只用"正常样本"训练，输出图像级分数 + 像素级热力图；
- 适合"缺陷样本稀少、类型未知"的产线场景，与 Frangi 形成**可解释基线 vs 深度学习**的对比。

注意：anomalib 依赖 torch / lightning，体积大。本模块**懒加载**，未安装时不影响项目运行。
"""
from __future__ import annotations

import os
import shutil

# ---- 数据布局转换 -----------------------------------------------------------


def build_mvtec_layout(good_images, crack_images, crack_masks, dst_root,
                       category: str = "mt_crack") -> dict:
    """把项目现有数据整理成 MVTec-AD 风格目录，供 anomalib 直接训练/评测。

    生成结构::

        dst_root/<category>/train/good/*.jpg
        dst_root/<category>/test/crack/*.jpg
        dst_root/<category>/ground_truth/crack/*.png

    参数均为 list[str]（图片绝对路径）。返回统计信息 dict。
    """
    cat = os.path.join(dst_root, category)
    d_train = os.path.join(cat, "train", "good")
    d_test = os.path.join(cat, "test", "crack")
    d_gt = os.path.join(cat, "ground_truth", "crack")
    for d in (d_train, d_test, d_gt):
        os.makedirs(d, exist_ok=True)

    n = {"train/good": 0, "test/crack": 0, "ground_truth/crack": 0}
    for src, dst, key in ((good_images, d_train, "train/good"),
                          (crack_images, d_test, "test/crack"),
                          (crack_masks, d_gt, "ground_truth/crack")):
        for p in src or []:
            if not os.path.isfile(p):
                continue
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
            n[key] += 1
    return {"root": cat, **n}


# ---- 推理 -------------------------------------------------------------------


def predict_patchcore(image_paths, model_dir=None, category: str = "mt_crack"):
    """用 PatchCore 推理；未训练时会提示先训练。

    返回 list[dict]：{"image": 路径, "score": 异常分数(越大越异常), "pred": 0/1}
    """
    try:
        from anomalib.deploy import OpenVINOInferencer  # noqa: F401  仅探测可用性
    except Exception:
        pass
    try:
        import torch
        from anomalib.models import Patchcore
    except Exception as e:                            # pragma: no cover
        raise RuntimeError(
            "anomalib 未安装。请运行 scripts/install_algo.bat，"
            "或 pip install -e ./vendor/anomalib"
        ) from e

    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError(
            "未找到已训练的 PatchCore 模型目录。先训练：\n"
            "  python scripts/train_patchcore.py\n"
            f"（默认输出 ./runs/{category}/PatchCore）"
        )

    from anomalib.data import Folder
    from anomalib.engine import Engine

    dm = Folder(root=str(model_dir), normal_dir="train/good",
                abnormal_dir="test/crack", mask_dir="ground_truth/crack",
                task="segmentation", image_size=(256, 256))
    engine = Engine(task="segmentation")
    model = Patchcore()
    results = engine.predict(model=model, datamodule=dm,
                             ckpt_path=os.path.join(str(model_dir), "weights", "model.ckpt"))
    out = []
    for batch in results:
        for p, s, pr in zip(getattr(batch, "image_path", []), batch.pred_score.tolist(),
                            batch.pred_label.tolist()):
            out.append({"image": p, "score": round(float(s), 4), "pred": int(pr)})
    return out


# ======================= 业务集成（供 views_algo 调用） =======================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_dataset_paths() -> dict:
    """定位项目里的裂纹/正常样本（datasets/vision/mt）。"""
    import glob

    base = os.path.join(PROJECT_ROOT, "datasets", "vision", "mt")
    return {
        "good": sorted(glob.glob(os.path.join(base, "free", "images", "*"))),
        "crack": sorted(glob.glob(os.path.join(base, "crack", "images", "*"))),
        "masks": sorted(glob.glob(os.path.join(base, "crack", "masks", "*"))),
    }


def build_dataset_from_project(dst_root: str | None = None, category: str = "mt_crack") -> dict:
    """把项目数据整理成 MVTec-AD 布局（供 anomalib 训练/评测）。"""
    paths = collect_dataset_paths()
    if not paths["good"] or not paths["crack"]:
        raise RuntimeError("未找到裂纹/正常样本，请检查 datasets/vision/mt")
    dst = dst_root or os.path.join(PROJECT_ROOT, "datasets", "auto")
    return build_mvtec_layout(paths["good"], paths["crack"], paths["masks"], dst, category)


def anomalib_available() -> bool:
    try:
        import anomalib  # noqa: F401
        return True
    except Exception:
        return False
