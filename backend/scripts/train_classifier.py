"""Train the lung-nodule binary classifier.

DO NOT RUN a full training run in this environment (no CUDA GPU, per the project
ground rules). This script is written as a clean, runnable CLI module intended
for Colab GPU. It CAN be smoke-tested locally on CPU via:

    python scripts/train_classifier.py --dry-run --epochs 1 --limit 32

A dry run exercises data loader -> model -> loss -> backward for a tiny subset
and exits without saving weights.

Hyperparameters (fixed spec):
    optimizer  : Adam lr=1e-3
    batch size : 16
    epochs     : 10
    loss       : CrossEntropyLoss (class-weighted from the manifest balance)
    scheduler  : ReduceLROnPlateau on val loss
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Allow running as `python scripts/train_classifier.py` from backend/ or repo root.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.datasets.classification_dataset import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    LABELS_CSV,
    LungClassificationDataset,
)
from app.models.classifier import LungClassifier

logger = logging.getLogger(__name__)


def class_weights() -> torch.Tensor:
    """Compute inverse-frequency class weights from the manifest balance."""
    import pandas as pd

    df = pd.read_csv(LABELS_CSV)
    counts = df["label"].map(LABEL_TO_INT).value_counts()
    total = counts.sum()
    weights = torch.ones(2, dtype=torch.float32)
    for cls, cnt in counts.items():
        weights[int(cls)] = total / (2 * cnt)
    return weights


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = torch.device(settings.device)
    logger.info("Device: %s", device)

    model = LungClassifier(use_pretrained=settings.use_pretrained).to(device)

    tr_ds: LungClassificationDataset | Subset = LungClassificationDataset(
        split="train", random_flip=True
    )
    val_ds: LungClassificationDataset | Subset = LungClassificationDataset(
        split="val", random_flip=False
    )

    if args.limit:
        tr_ds = Subset(tr_ds, list(range(min(args.limit, len(tr_ds)))))
        val_ds = Subset(val_ds, list(range(min(max(1, args.limit // 4), len(val_ds)))))

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    weights = class_weights().to(device)
    logger.info("Class weights: %s", weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_clf = 0.0
        n_batches = 0
        start = time.time()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_clf += loss.item()
            n_batches += 1
            if args.dry_run and n_batches >= args.dry_run_batches:
                break

        avg_train = running_clf / max(1, n_batches)
        logger.info(
            "Epoch %d/%d train_loss=%.4f acc=%.4f time=%.1fs",
            epoch,
            args.epochs,
            avg_train,
            0.0,
            time.time() - start,
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
                val_batches += 1
                if args.dry_run and val_batches >= max(1, args.dry_run_batches):
                    break
        val_loss /= max(1, val_batches)
        val_acc = correct / max(1, total)
        scheduler.step(val_loss)
        logger.info("          val_loss=%.4f val_acc=%.4f", val_loss, val_acc)

        if args.dry_run:
            break

        if val_loss < best_val_loss and not args.dry_run:
            best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                args.output,
            )
            logger.info("Saved best checkpoint to %s", args.output)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Train the lung classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=settings.classifier_weights,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny subset for 1 epoch (CPU smoke test) and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per split (used with --dry-run).",
    )
    parser.add_argument(
        "--dry-run-batches", type=int, default=2, help="Batches for dry-run."
    )
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = min(args.epochs, 1)
        args.limit = args.limit if args.limit is not None else 32
        logger.info("DRY RUN mode (CPU smoke test).")
    elif args.dry_run and args.limit is None:
        args.limit = 32

    train(args)


if __name__ == "__main__":
    main()
