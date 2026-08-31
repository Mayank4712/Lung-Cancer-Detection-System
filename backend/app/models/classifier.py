"""Lung-nodule binary classifier (ResNet-50 backbone, grayscale-adapted).

Architecture:
    resnet50 backbone
    conv1  : nn.Conv2d(1, 64, k=7, s=2, p=3) -- grayscale adaptation
    layers : standard resnet stages (layer4[-1] is the GradCAM target layer)
    head   : Linear(2048,512) -> BN -> ReLU -> Dropout(0.5)
             -> Linear(512,128) -> ReLU -> Dropout(0.3) -> Linear(128,2)
    output : raw logits (no final activation)

Input contract: (N, 1, 256, 256) float tensor normalized to mean/std 0.5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LungClassifier(nn.Module):
    def __init__(self, use_pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        )
        self.backbone = models.resnet50(weights=weights)

        # Grayscale adaptation: replace conv1's 3 input channels with 1, keeping
        # the pretrained values averaged across the channel dimension.
        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if use_pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        self.backbone.conv1 = new_conv1

        # Replace the fc head.
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        if use_pretrained:
            # Note: backbone is left trainable by default; call
            # freeze_backbone() in the training script for a freeze+fine-tune
            # schedule as needed.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.backbone(x)

    def freeze_backbone(self, freeze: bool) -> None:
        """Freeze the backbone (leave the custom head trainable) when freeze=True.

        When freeze=False, unfreeze everything.
        """
        for name, param in self.backbone.named_parameters():
            if name.startswith("fc."):
                # Head stays trainable regardless; only the backbone gets frozen.
                param.requires_grad = True
            else:
                param.requires_grad = not freeze

    def target_layer(self) -> nn.Module:
        """GradCAM target layer for the classifier."""
        return self.backbone.layer4[-1]
