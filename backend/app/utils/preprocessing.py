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

# Fraction of the image margin to neutralize on each side. The trained
# classifier leaned on a class-correlated border/corner shortcut (dark borders
# on Healthy vs bright borders on Lung_Nodule). Neutralizing the outer margin
# forces the model to rely on internal lung-tissue features instead.
BORDER_MARGIN = 0.15


def to_grayscale_float(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def central_roi_mask(shape: tuple[int, int], margin: float = BORDER_MARGIN) -> np.ndarray:
    """Boolean ROI mask that keeps the central region and discards the outer margin.

    Returns a (H, W) bool array, True inside a central ellipse inscribed in the
    rectangle that excludes ``margin`` on each side (sized off the smaller
    dimension, so the mask is identical for square letterboxed inputs regardless
    of aspect). Used consistently by both the classifier-input neutralization and
    the GradCAM heatmap masking.
    """
    h, w = shape
    min_dim = min(h, w)
    keep = 1.0 - 2.0 * margin
    rx = max(1.0, (keep * min_dim) / 2.0)
    ry = max(1.0, (keep * min_dim) / 2.0)
    cy, cx = h / 2.0, w / 2.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask


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
    """Return a (1, 256, 256) float32 tensor normalized to mean/std 0.5.

    The outer margin (see BORDER_MARGIN) is neutralized so the classifier cannot
    exploit the class-correlated border/corner brightness shortcut and must rely
    on internal lung-tissue features.
    """
    img = _letterbox(img.convert("L"), CLF_SIZE)
    arr = to_grayscale_float(img)
    arr = (arr - MEAN) / STD
    mask = central_roi_mask(arr.shape[:2], margin=BORDER_MARGIN)
    arr = arr * mask  # zero the outer margin (0 == neutral 0.5 grayscale)
    return arr[None, :, :]  # (1, H, W)


def preprocess_segmentation(img: Image.Image) -> np.ndarray:
    """Return a (1, 512, 512) float32 tensor normalized to mean/std 0.5."""
    img = _center_crop_square(img.convert("L")).resize(
        (SEG_SIZE, SEG_SIZE), Image.BILINEAR
    )
    arr = to_grayscale_float(img)
    arr = (arr - MEAN) / STD
    return arr[None, :, :]  # (1, H, W)
