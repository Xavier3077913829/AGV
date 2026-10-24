# DeepCrack 集成说明

这个目录是从官方 DeepSegmentor 抽取出来的 DeepCrack 算法，已经去掉了旧框架依赖，只保留 PyTorch 核心，方便直接放进你们的项目。

## 文件

- `deepcrack.py`：网络结构 + `CrackSegmenter` 推理封装，单文件，可单独拷贝进项目。
- `train.py`：精简的训练脚本，适配 DeepCrack 数据集目录格式。

## 依赖

- Python 3.8+
- PyTorch
- NumPy、Pillow

## 推理

```python
from deepcrack import CrackSegmenter

# weights_path 可以是官方 pretrained_net_G.pth，也可以是自己训练的权重
segmenter = CrackSegmenter(weights_path="pretrained_net_G.pth", device="cpu")

prob = segmenter.predict("crack.jpg")       # (H, W) float32，0到1的裂纹概率
mask = segmenter.predict_mask("crack.jpg")  # (H, W) uint8，0或255
```

`predict` 接受 PIL 图像、RGB 的 NumPy 数组或图片路径。输入请用 RGB 顺序。

## 训练

数据按下面的目录放好，`train_lab` 里的标注是 0/255 的二值 PNG：

```text
DeepCrack/
    train_img/
    train_lab/
    test_img/
    test_lab/
```

```bash
python train.py --data_root /path/to/DeepCrack --save_path deepcrack.pth --epochs 200
```

常用参数：`--batch_size`、`--image_size`、`--lr`、`--step_every`、`--val_every`、`--device`。

## 官方预训练权重

官方 README 提供了 DeepCrack 预训练模型，Google Drive 和百度网盘链接在官方仓库 [yhlleo/DeepSegmentor](https://github.com/yhlleo/DeepSegmentor) 里。下载后把 `.pth` 路径传给 `CrackSegmenter` 即可。

本目录的模型属性名与官方 checkpoint 保持一致，官方 `pretrained_net_G.pth` 可以直接加载。

## 说明

- 网络是全卷积的，任意尺寸都能推理，但官方权重在 256×256 和 544×384 附近训练，尺寸差异太大时结果会退化。
- 输出是单通道 logit，最终结果用 sigmoid 转成概率。评估时建议用 0.5 阈值计算 IoU / F1，并和 Frangi 基线对比。
