"""Input preprocessing for inference.

Converts an uploaded PIL image into the tensors expected by the classifier and
the segmenter, mirroring the training-time transforms so predictions are
consistent with what the models were trained on:
    classifier : single-channel 256x256, letterbox-resized (aspect preserved),
                 normalized to mean/std 0.5
    segmenter  : single-channel, kept at original aspect via center-crop to a
                 square then resize; padded to a multiple of 32 inside the model
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CLF_SIZE = 256
SEG_SIZE = 512
MEAN = 0.5
STD = 0.5


def to_grayscale_float(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    src_w, src_h = image.size
    scale = min(size / src_w, size / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return image.crop((left, top, left + s, top + s))


def preprocess_classifier(img: Image.Image) -> np.ndarray:
    """Return a (1, 256, 256) float32 tensor normalized to mean/std 0.5."""
    img = _letterbox(img.convert("L"), CLF_SIZE)
    arr = to_grayscale_float(img)
    arr = (arr - MEAN) / STD
    return arr[None, :, :]  # (1, H, W)


def preprocess_segmentation(img: Image.Image) -> np.ndarray:
    """Return a (1, 512, 512) float32 tensor normalized to mean/std 0.5."""
    img = _center_crop_square(img.convert("L")).resize(
        (SEG_SIZE, SEG_SIZE), Image.BILINEAR
    )
    arr = to_grayscale_float(img)
    arr = (arr - MEAN) / STD
    return arr[None, :, :]  # (1, H, W)
