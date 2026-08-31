"""Classification dataset.

Reads data/labels.csv (the manifest built by scripts/build_labels.py) and serves
(256, 256) single-channel normalized tensors with integer labels.

A fixed 70/15/15 train/val/test split (random_state=42) is computed once and
cached to data/splits.json so every run across processes re-uses the identical
partition (reproducible without recomputing).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from app.config import settings

# Conservative split that keeps at least one sample of every class in val/test
# even for tiny minority classes.
HEALTHY = "Healthy"
LUNG_NODULE = "Lung_Nodule"
LABEL_TO_INT = {HEALTHY: 0, LUNG_NODULE: 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
LABELS_CSV = DATA_DIR / "labels.csv"
SPLITS_JSON = DATA_DIR / "splits.json"

IMG_SIZE = 256
MEAN = 0.5
STD = 0.5


def _load_splits(df: pd.DataFrame) -> dict[str, list[int]]:
    """Load cached splits or compute the stratified 70/15/15 split and cache it."""
    if SPLITS_JSON.exists():
        with open(SPLITS_JSON, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        n = len(df)
        if (
            set(cached.keys()) == {"train", "val", "test"}
            and len(cached["train"]) + len(cached["val"]) + len(cached["test"]) == n
        ):
            return cached

    labels = df["label"].map(LABEL_TO_INT).to_numpy()
    idx = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        idx, test_size=0.15, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=labels[train_val_idx],
    )

    splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_JSON, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    return splits


class LungClassificationDataset(Dataset):
    """(image_tensor, label_int) pairs for 70/15/15 reproducible splits.

    Images are stored/loaded as single-channel grayscale. Each sample is
    letterbox-resized to 256x256 (aspect ratio preserved, padded with zero) and
    normalized to mean/std 0.5. A light RandomHorizontalFlip(p=0.3) is applied on
    the train split only.
    """

    def __init__(
        self,
        split: str = "train",
        labels_csv: Optional[Path] = None,
        random_flip: bool = True,
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
        self.random_flip = random_flip and split == "train"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        row = self.label_csv.iloc[self.indices[idx]]
        image = Image.open(row["image_path"]).convert("L")
        image = _letterbox_resize(image, IMG_SIZE, IMG_SIZE)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        tensor = arr[None, :, :]  # (1, H, W)

        if self.random_flip and np.random.rand() < 0.3:
            tensor = tensor[:, :, ::-1].copy()

        label = LABEL_TO_INT[row["label"]]
        return tensor, label


def _letterbox_resize(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize preserving aspect ratio and pad (zero-fill) to the target box."""
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (target_w, target_h), 0)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas
