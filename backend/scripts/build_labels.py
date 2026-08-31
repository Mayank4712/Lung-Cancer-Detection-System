"""One-time manifest builder.

Walks the raw paired dataset, derives the binary classification label from each
mask, and writes data/labels.csv (filename,label,mask_path,image_path).

Usage:
    python -m scripts.build_labels [--dataset-root PATH] [--output PATH]

This is a one-time setup step; it should be re-run only after the raw dataset is
modified. The output manifest is consumed by both dataset classes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_labels.py` from the backend/ dir or from
# the repo root.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.datasets.labels import build_manifest

DEFAULT_DATASET_ROOT = BACKEND_DIR.parent / "Lung_Nodule_Dataset" / "Lung_Nodule_Dataset"
DEFAULT_OUTPUT = BACKEND_DIR.parent / "data" / "labels.csv"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build the label manifest.")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the paired dataset root (contains Images + Annotations dirs).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write data/labels.csv.",
    )
    args = parser.parse_args()

    df = build_manifest(Path(args.dataset_root), Path(args.output))
    print(f"\nManifest written to {args.output} ({len(df)} rows).")


if __name__ == "__main__":
    main()
