"""Segmentation dataset.

Serves (image_tensor, mask_tensor) pairs for the lung-nodule segmenter.

The dataset shares the same manifest (data/labels.csv) and the same fixed
train/val/test split cache (data/splits.json) as the classification dataset, so
the classifier and segmenter always train/evaluate on identical partitions.

Design decision (documented): masks are resized to a fixed 512x512. This trades
original-resolution fidelity for (a) uniform, deterministic tensor shapes that a
batch collator can stack without padding hacks, and (b) a memory-stable size for
GPU training on Colab. 512x512 is well above the ~nodule region sizes in this
dataset (nodules are typically tens of pixels across) so it preserves diagnostic
detail while bounding VRAM. Masks are binarized at 0.5 and normalized consistently
with the classifier (mean/std 0.5). Per the medical-accuracy requirement, only
resize + normalize are applied — no random flips/crops.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from app.datasets.classification_dataset import _load_splits, LABELS_CSV

IMG_SIZE = 512
MEAN = 0.5
STD = 0.5


def _resize_with_objects(seed: int):
    """Deterministic per-seed RNG so transforms are reproducible if re-run."""
    return random.Random(seed)


class LungSegmentationDataset(Dataset):
    """(image_tensor, mask_tensor) pairs at fixed 512x512 resolution.

    Both image and mask are resized to 512x512 and normalized to mean/std 0.5.
    Masks are binarized at 0.5 (values >= 0.5 -> 1). Roughly square-frame images
    in this dataset make a non-padding resize acceptable; for strongly non-square
    inputs, center-crop is applied before resize to keep object geometry intact.
    """

    def __init__(
        self,
        split: str = "train",
        labels_csv: Optional[Path] = None,
        size: int = IMG_SIZE,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        labels_csv = labels_csv or LABELS_CSV
        if not labels_csv.exists():
            raise FileNotFoundError(
                f"{labels_csv} not found. Run: python scripts/build_labels.py"
            )
        df = pd.read_csv(labels_csv)
        splits = _load_splits(df)
        self.indices = splits[split]
        self.label_csv = df
        self.size = size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.label_csv.iloc[self.indices[idx]]
        image = Image.open(row["image_path"]).convert("L")
        mask = Image.open(row["mask_path"]).convert("L")

        image, mask = _resize_pair(image, mask, self.size)

        img = np.asarray(image, dtype=np.float32) / 255.0
        img = (img - MEAN) / STD
        img_tensor = img[None, :, :]  # (1, H, W)

        msk = np.asarray(mask, dtype=np.float32) / 255.0
        mask_bin = (msk >= 0.5).astype(np.float32)
        mask_tensor = mask_bin[None, :, :]  # (1, H, W)

        return img_tensor, mask_tensor


def _resize_pair(
    image: Image.Image, mask: Image.Image, size: int
) -> tuple[Image.Image, Image.Image]:
    """Center-crop to square (if needed) then resize both to (size, size)."""
    w, h = image.size
    if w != h:
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        box = (left, top, left + s, top + s)
        image = image.crop(box)
        mask = mask.crop(box)
    image = image.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)
    return image, mask
