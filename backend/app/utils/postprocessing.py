"""Output post-processing for inference.

Turns raw model outputs into API-ready payloads:
    - segmentation logits -> binary mask PNG (base64), confidence map PNG,
      bounding box, and nodule pixel area
    - GradCAM CAM -> colorized heatmap PNG and blended overlay PNG (base64)
All PNG encodings use cv2.imencode and are returned as "data:image/png;base64,..."
Data URIs consumed directly by the frontend.
"""
from __future__ import annotations

import base64
from typing import Optional, Tuple

import cv2
import numpy as np


def _encode_bgr_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def process_segmentation(
    logits: np.ndarray,
    confidence_threshold: float = 0.5,
) -> dict:
    """Convert (1, H, W) raw logits into the segmentation result dict.

    Omission policy (documented in schemas.py):
        When the predicted mask is empty and no bounding box can be found, all
        four bounding_box coordinates are set to None and has_nodule is False.
        This is deterministic and easy for the frontend to branch on.
    """
    # Accept either (H, W) or (1, H, W) logits; normalize to 2D (H, W).
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]

    mask = (sigmoid(logits) > confidence_threshold).astype(np.uint8)
    has_nodule = bool(mask.sum() > 0)
    area_pixels = int(mask.sum()) if has_nodule else 0

    bbox: Optional[Tuple[int, int, int, int]] = None
    if has_nodule:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r_min, r_max = np.argmax(rows), mask.shape[0] - np.argmax(np.flipud(rows)) - 1
        c_min, c_max = np.argmax(cols), mask.shape[1] - np.argmax(np.flipud(cols)) - 1
        bbox = (int(c_min), int(r_min), int(c_max), int(r_max))

    # Binary (black/white) mask visualization.
    mask_img = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

    # Confidence map: normalize the sigmoid probs to [0, 255].
    probs = sigmoid(logits)
    p_min, p_max = float(probs.min()), float(probs.max())
    if p_max > p_min:
        norm = (probs - p_min) / (p_max - p_min)
    else:
        norm = np.zeros_like(probs)
    conf_img = (norm * 255).astype(np.uint8)
    conf_img = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
    conf_img = cv2.cvtColor(conf_img, cv2.COLOR_BGR2RGB)

    return {
        "has_nodule": has_nodule,
        "mask_base64": _encode_bgr_png(mask_img),
        "bounding_box": (
            {
                "x_min": bbox[0],
                "y_min": bbox[1],
                "x_max": bbox[2],
                "y_max": bbox[3],
            }
            if bbox is not None
            else {"x_min": None, "y_min": None, "x_max": None, "y_max": None}
        ),
        "nodule_area_pixels": area_pixels,
        "confidence_map": _encode_bgr_png(cv2.cvtColor(conf_img, cv2.COLOR_RGB2BGR)),
    }


def process_gradcam(
    cam: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> dict:
    """Build heatmap + overlay PNG Data URIs from a [0,1] CAM.

    The overlay uses the CAM itself as a per-pixel blend weight, so zero-intensity
    (background) regions remain untouched while active regions take on the JET
    color — no constant-alpha tint leaks over the whole slice.
    """
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    cam_uint8 = (cam * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cam_uint8, colormap)  # BGR

    # Heatmap alone (as RGB PNG).
    heatmap_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    heatmap_bgr_for_encode = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    # Overlay: wrap the colored heatmap to the original size when needed.
    orig = np.asarray(original_rgb)
    if colored.shape[:2] != orig.shape[:2]:
        colored = cv2.resize(
            colored, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        blur_cam = cv2.resize(
            cam, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    else:
        blur_cam = cam

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    # Per-pixel alpha weight: 0 where the CAM is 0 (hide), else scaled by the
    # constant alpha for a clean, focused blend.
    per_pixel_alpha = (blur_cam[..., None].astype(np.float32)) * np.float32(alpha)
    overlay_img = (
        orig_bgr.astype(np.float32) * (1.0 - per_pixel_alpha)
        + colored.astype(np.float32) * per_pixel_alpha
    ).astype(np.uint8)

    return {
        "heatmap_base64": _encode_bgr_png(heatmap_bgr_for_encode),
        "overlay_base64": _encode_bgr_png(overlay_img),
    }
