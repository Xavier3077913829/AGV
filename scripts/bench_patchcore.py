# -*- coding: utf-8 -*-
"""基准实验 3：anomalib PatchCore 评测（图像级 AUROC / F1）

用法::  python scripts/bench_patchcore.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    data_root = os.path.join(ROOT, "datasets", "auto", "mt_crack")
    if not os.path.isdir(data_root):
        sys.exit("先运行 scripts/prepare_anomalib_dataset.py")
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except Exception as e:
        sys.exit(f"[需要安装] anomalib：{e}")

    dm = Folder(root=data_root, normal_dir="train/good", abnormal_dir="test/crack",
                mask_dir="ground_truth/crack", task="segmentation", image_size=(256, 256))
    model = Patchcore(backbone="resnet18")
    engine = Engine(task="segmentation", default_root_dir=os.path.join(ROOT, "runs"))
    ckpt = os.path.join(ROOT, "runs", "Patchcore", "weights", "model.ckpt")
    if not os.path.isfile(ckpt):
        sys.exit(f"未找到权重 {ckpt}\n先运行: python scripts/train_patchcore.py")
    res = engine.test(model=model, datamodule=dm, ckpt_path=ckpt)
    print("\n=== PatchCore 测试结果 ===")
    for k, v in (res[0] if isinstance(res, list) else res).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("\n报告写法示例：在自建裂纹数据集上，PatchCore 图像级 AUROC / F1 见上表，"
          "与 Frangi 黑脊基线（传统算法）形成互补。")


if __name__ == "__main__":
    main()
