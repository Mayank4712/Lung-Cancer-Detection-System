"""Switch active model weights to a saved version.

Usage:
    python scripts/switch_weights.py v1_original
    python scripts/switch_weights.py v2_regularized

Copies the version's classifier.pth and segmenter.pth into the active
weights directory (app/weights/), overwriting whatever is there.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "app" / "weights" / "versions"
ACTIVE_DIR = Path(__file__).resolve().parent.parent / "app" / "weights"
WEIGHT_FILES = ("classifier.pth", "segmenter.pth")


def main(version: str) -> None:
    version_dir = VERSIONS_DIR / version
    if not version_dir.is_dir():
        available = [d.name for d in VERSIONS_DIR.iterdir() if d.is_dir()]
        print(f"Error: version '{version}' not found in {VERSIONS_DIR}")
        print(f"Available versions: {available or '(none)'}")
        sys.exit(1)

    for fname in WEIGHT_FILES:
        src = version_dir / fname
        dst = ACTIVE_DIR / fname
        if not src.exists():
            print(f"Error: {src} does not exist")
            sys.exit(1)
        shutil.copy2(src, dst)
        print(f"  Copied {fname} ({src.stat().st_size:,} bytes)")

    print(f"\nActive weights switched to '{version}'.")
    print("Restart the server for changes to take effect.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/switch_weights.py <version_name>")
        print("Example: python scripts/switch_weights.py v1_original")
        sys.exit(1)
    main(sys.argv[1])
