# -*- coding: utf-8 -*-
"""训练 PatchCore（anomalib）—— 供 bench/推理使用

用法::  python scripts/train_patchcore.py
前置::  python scripts/prepare_anomalib_dataset.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    data_root = os.path.join(ROOT, "datasets", "auto", "mt_crack")
    if not os.path.isdir(data_root):
        sys.exit("先运行 scripts/prepare_anomalib_dataset.py 生成数据集")

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except Exception as e:
        sys.exit(f"[需要安装] anomalib：{e}")

    dm = Folder(root=data_root, normal_dir="train/good", abnormal_dir="test/crack",
                mask_dir="ground_truth/crack", task="segmentation", image_size=(256, 256))
    model = Patchcore(backbone="resnet18", coreset_sampling_ratio=0.1)
    engine = Engine(task="segmentation", max_epochs=1, default_root_dir=os.path.join(ROOT, "runs"))
    engine.fit(model=model, datamodule=dm)
    print("训练完成，模型在 runs/ 下。")
    print("评测：  python scripts/bench_patchcore.py")


if __name__ == "__main__":
    main()
