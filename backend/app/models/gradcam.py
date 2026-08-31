"""Generic GradCAM implementation.

Registers a forward hook (activations) and a full backward hook (gradients) on a
target layer, computes the class-activation map, and provides colorize / overlay
helpers for the post-processing pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute the CAM for ``input_tensor`` and return a [0, 1] float map.

        input_tensor: (1, C, H, W). Returns (H, W) float32 normalized to [0, 1],
        bilinearly resized to the input spatial size.
        """
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        logits = self.model(input_tensor)

        if isinstance(logits, dict):  # segmenter-style dict losing channel
            logits = logits["out"]

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        target_score = logits[0, target_class]

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        cam = cam.squeeze(0).squeeze(0)  # (H', W')
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_tensor.shape[-2], input_tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        cam = F.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def __del__(self) -> None:
        try:
            self._fh.remove()
            self._bh.remove()
        except Exception:
            pass


def colorize(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a [0, 1] CAM to a uint8 BGR (from OpenCV) colormap image."""
    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cam_uint8, colormap)
    return colored


def overlay(
    original_rgb: np.ndarray, colored_cam: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Blend ``colored_cam`` over ``original_rgb`` and return BGR.

    original_rgb: HxWx3 uint8 RGB image. Returns HxWx3 uint8 BGR suitable for
    cv2.imencode.
    """
    if colored_cam.shape[:2] != original_rgb.shape[:2]:
        colored_cam = cv2.resize(
            colored_cam, (original_rgb.shape[1], original_rgb.shape[0])
        )
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, colored_cam, alpha, 0)
    return blended
