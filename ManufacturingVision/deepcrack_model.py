"""Self-contained DeepCrack model for crack segmentation.

Reference:
    Liu, Y., Yao, J., Lu, X., Xie, R., Li, L.
    "DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack
    Segmentation." Neurocomputing, 338:139-153, 2019.

This file has no dependency on the original DeepSegmentor codebase and can be
dropped directly into a PyTorch project. The network is fully convolutional,
so it accepts any input size, although the official weights were trained
around 256x256 and 544x384 inputs.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _get_norm_layer(norm: str):
    if norm == "batch":
        return nn.BatchNorm2d
    if norm == "instance":
        return nn.InstanceNorm2d
    if norm == "none":
        return None
    raise ValueError(f"unknown normalization layer: {norm}")


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer, num_blocks):
        layers = []
        for i in range(num_blocks):
            current_in = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(
                    current_in,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(out_channels))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DeepCrackNet(nn.Module):
    """VGG-style encoder with five side outputs and one fused output."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_filters: int = 64,
        norm: str = "batch",
    ):
        super().__init__()
        norm_layer = _get_norm_layer(norm)

        # Attribute names must match the official checkpoint keys.
        self.conv1 = _ConvBlock(in_channels, base_filters, norm_layer, 2)
        self.conv2 = _ConvBlock(base_filters, base_filters * 2, norm_layer, 2)
        self.conv3 = _ConvBlock(base_filters * 2, base_filters * 4, norm_layer, 3)
        self.conv4 = _ConvBlock(base_filters * 4, base_filters * 8, norm_layer, 3)
        self.conv5 = _ConvBlock(base_filters * 8, base_filters * 8, norm_layer, 3)

        self.side_conv1 = nn.Conv2d(base_filters, num_classes, kernel_size=1, bias=False)
        self.side_conv2 = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1, bias=False)
        self.side_conv3 = nn.Conv2d(base_filters * 4, num_classes, kernel_size=1, bias=False)
        self.side_conv4 = nn.Conv2d(base_filters * 8, num_classes, kernel_size=1, bias=False)
        self.side_conv5 = nn.Conv2d(base_filters * 8, num_classes, kernel_size=1, bias=False)

        self.fuse_conv = nn.Conv2d(num_classes * 5, num_classes, kernel_size=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        h, w = x.shape[2:]
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))

        side1 = self.side_conv1(conv1)
        side2 = self._upsample(self.side_conv2(conv2), h, w)
        side3 = self._upsample(self.side_conv3(conv3), h, w)
        side4 = self._upsample(self.side_conv4(conv4), h, w)
        side5 = self._upsample(self.side_conv5(conv5), h, w)

        fused = self.fuse_conv(torch.cat([side1, side2, side3, side4, side5], dim=1))
        # The last return value is the fused output used as the final prediction.
        return side1, side2, side3, side4, side5, fused

    @staticmethod
    def _upsample(x, h, w):
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)


def create_deepcrack(
    in_channels: int = 3,
    num_classes: int = 1,
    base_filters: int = 64,
    norm: str = "batch",
):
    return DeepCrackNet(in_channels, num_classes, base_filters, norm)


class CrackSegmenter:
    """Load a DeepCrack checkpoint and run crack-mask inference."""

    def __init__(
        self,
        weights_path: Optional[Union[str, os.PathLike]] = None,
        num_classes: int = 1,
        base_filters: int = 64,
        norm: str = "batch",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = DeepCrackNet(3, num_classes, base_filters, norm).to(self.device)
        if weights_path is not None:
            self.load_weights(weights_path)
        self.net.eval()

    def load_weights(self, weights_path: Union[str, os.PathLike]) -> None:
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "netG" in state:
                state = state["netG"]
            state = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in state.items()
            }
        try:
            self.net.load_state_dict(state)
        except RuntimeError:
            # Checkpoints saved by older PyTorch may lack BatchNorm
            # num_batches_tracked counters.
            result = self.net.load_state_dict(state, strict=False)
            print(
                "warning: weights loaded non-strictly "
                f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
            )

    @torch.no_grad()
    def predict(self, image) -> np.ndarray:
        """Return the fused crack probability map, shape (H, W), float32 in [0, 1]."""
        tensor = self._preprocess(image)
        fused = self.net(tensor)[-1]
        return torch.sigmoid(fused).squeeze(0).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_mask(self, image, threshold: float = 0.5) -> np.ndarray:
        """Return a binarized crack mask, shape (H, W), uint8 0 or 255."""
        prob = self.predict(image)
        return (prob >= threshold).astype(np.uint8) * 255

    def _preprocess(self, image) -> torch.Tensor:
        if Image is not None and isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        elif isinstance(image, (str, os.PathLike)):
            if Image is None:
                raise RuntimeError("Pillow is required to load image paths")
            image = np.asarray(Image.open(image).convert("RGB"))

        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxW or HxWx3 RGB image, got {array.shape}")

        array = array.astype(np.float32) / 255.0
        array = (array - _MEAN) / _STD
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
