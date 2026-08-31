"""Label-manifest builder.

Walks the raw paired dataset (Images + Annotations folders) once, derives the
binary classification label from each mask (all-blank -> Healthy, any foreground
pixel -> Lung_Nodule), and writes a single manifest to data/labels.csv so both
the classifier and segmenter datasets read one canonical source instead of
re-deriving labels on every run.

Called on demand by scripts/build_labels.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

HEALTHY = "Healthy"
LUNG_NODULE = "Lung_Nodule"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Suffixes that mark an annotation/mask file and should be stripped when
# matching an image to its annotation by filename stem.
MASK_SUFFIXES = ("_mask", "_ann", "_annotation", "_label", "_gt", "_ground_truth")


def _strip_mask_suffix(stem: str) -> str:
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: len(stem) - len(suffix)]
    return stem


def _locate_subfolders(dataset_root: Path):
    """Find the images folder and the annotations/masks folder under ``dataset_root``.

    Logs what was actually found so the pairing logic is transparent rather than
    assumed. Returns (images_dir, annotations_dir).
    """
    csv_dir = dataset_root
    images_dir: Optional[Path] = None
    ann_dir: Optional[Path] = None

    for candidate in ("images", "Images", "image", "Image", "imgs"):
        p = dataset_root / candidate
        if p.is_dir():
            images_dir = p
            break
    for candidate in (
        "annotations",
        "Annotations",
        "masks",
        "Masks",
        "annotation",
        "mask",
    ):
        p = dataset_root / candidate
        if p.is_dir():
            ann_dir = p
            break

    # If not found at the top level, recurse one level deeper (the real payload
    # often sits one directory below the archive name, e.g. data/ds/images).
    if images_dir is None or ann_dir is None:
        for sub in sorted(p for p in csv_dir.iterdir() if p.is_dir()):
            for candidate in ("images", "Images"):
                p = sub / candidate
                if p.is_dir() and images_dir is None:
                    images_dir = p
            for candidate in ("annotations", "Annotations", "masks", "Masks"):
                p = sub / candidate
                if p.is_dir() and ann_dir is None:
                    ann_dir = p

    if images_dir is None or ann_dir is None:
        raise FileNotFoundError(
            f"Could not locate Images and Annotations/Masks folders under {dataset_root}"
        )

    logger.info("Found images folder:      %s", images_dir)
    logger.info("Found annotations folder: %s", ann_dir)
    print(f"Found images folder:      {images_dir}")
    print(f"Found annotations folder: {ann_dir}")
    return images_dir, ann_dir


def _prefix_stem(stem: str) -> str:
    """Handle files that are prefixed rather than suffixed with '_mask'."""
    if stem.startswith("mask_"):
        return stem[len("mask_"):]
    if stem.startswith("mask"):
        return stem[len("mask"):]
    return stem


def build_manifest(dataset_root: Path, output_csv: Path) -> pd.DataFrame:
    """Walk the dataset, pair images with annotations, derive labels, save CSV.

    Returns the manifest DataFrame. Unpaired files are skipped and logged.
    """
    images_dir, ann_dir = _locate_subfolders(dataset_root)

    image_files: dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_files[p.stem.lower()] = p

    ann_files: dict[str, Path] = {}
    for p in ann_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            ann_files[_strip_mask_suffix(p.stem).lower()] = p

    rows: list[dict] = []
    skipped = 0
    for stem, img_path in image_files.items():
        ann_key = _prefix_stem(stem)
        ann_path = ann_files.get(ann_key) or ann_files.get(_strip_mask_suffix(stem))
        if ann_path is None:
            skipped += 1
            continue

        mask = np.array(Image.open(ann_path).convert("L"))
        has_nodule = bool(mask.any() and mask.sum() > 0)
        label = LUNG_NODULE if has_nodule else HEALTHY

        rows.append(
            {
                "filename": img_path.name,
                "image_path": str(img_path),
                "mask_path": str(ann_path),
                "label": label,
            }
        )

    if not rows:
        print("No paired image/annotation pairs were found.")
        return pd.DataFrame(columns=["filename", "image_path", "mask_path", "label"])

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Total paired samples: {len(df)}")
    print(f"Skipped (unpaired):   {skipped}")
    print("\nClass balance:")
    balance = df["label"].value_counts()
    for label, count in balance.items():
        print(f"  {label}: {count} ({count / len(df) * 100:.2f}%)")

    return df
