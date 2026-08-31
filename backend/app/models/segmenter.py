"""Lung-nodule binary segmenter (DeepLabV3-ResNet101 backbone, grayscale-adapted).

Architecture:
    torchvision deeplabv3_resnet101 backbone
    backbone.conv1 : nn.Conv2d(1, 64, k=7, s=2, p=3) -- grayscale adaptation
    classifier head: ASPP (256) -> conv 256->256 -> BN -> ReLU -> final
                     Conv2d(256, 1, 1) -- binary (1 channel) instead of 21 classes
    forward returns raw logits (N, 1, H, W); sigmoid is applied only at
    inference/post-processing time (required for BCEWithLogitsLoss during
    training).

Input contract: single-channel, arbitrary (H, W). Input is padded to a multiple
of 32 (DeepLab atrous receptive field / output stride) and cropped back after
forward so the output exactly matches the input spatial size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LungSegmenter(nn.Module):
    def __init__(self, use_pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            if use_pretrained
            else None
        )
        self.model = models.segmentation.deeplabv3_resnet101(weights=weights)
        self.backbone = self.model.backbone

        # Grayscale adaptation of the backbone's first conv.
        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if use_pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        self.backbone.conv1 = new_conv1

        # Replace the final projection so it outputs 1 channel (binary).
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        # Match the Colab checkpoint, which also made the aux head binary.
        if getattr(self.model, "aux_classifier", None) is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        orig_h, orig_w = x.shape[-2:]
        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32
        if pad_h or pad_w:
            x = nn.functional.pad(
                x, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        out = self.model(x)["out"]  # DeepLabV3 handles backbone->classifier wiring

        if pad_h or pad_w:
            out = out[:, :, :orig_h, :orig_w]

        return out  # (N, 1, H, W) raw logits

    def target_layer(self) -> nn.Module:
        """GradCAM target layer for the segmenter."""
        return self.backbone.layer4[-1]
