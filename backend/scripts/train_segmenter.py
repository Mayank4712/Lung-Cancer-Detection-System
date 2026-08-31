"""Train the lung-nodule binary segmenter.

DO NOT RUN a full training run in this environment (no CUDA GPU, per the project
ground rules). This script is a clean, runnable CLI module intended for Colab GPU.
It CAN be smoke-tested locally on CPU via:

    python scripts/train_segmenter.py --dry-run --epochs 1 --limit 32

A dry run exercises data loader -> model -> loss -> backward for a tiny subset
and exits without saving weights.

Hyperparameters (fixed spec):
    optimizer  : Adam lr=1e-4
    batch size : 8 (4 fallback for GPU memory)
    loss       : 0.5*BCEWithLogits + 0.5*Dice
    grad clip  : 1.0
    early stop : patience 10 on val Dice
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Allow running as `python scripts/train_segmenter.py` from backend/ or repo root.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.datasets.segmentation_dataset import LungSegmentationDataset
from app.models.segmenter import LungSegmenter

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """0.5 * BCEWithLogits + 0.5 * Dice loss on 1-channel logits."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        return 0.5 * bce + 0.5 * dice_loss


def dice_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(preds)
    pred_bin = (probs >= 0.5).float()
    smooth = 1.0
    inter = (pred_bin * targets).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return float(((2 * inter + smooth) / (union + smooth)).mean().item())


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

    model = LungSegmenter(use_pretrained=settings.use_pretrained).to(device)

    tr_ds: LungSegmentationDataset | Subset = LungSegmentationDataset(split="train")
    val_ds: LungSegmentationDataset | Subset = LungSegmentationDataset(split="val")

    if args.limit:
        tr_ds = Subset(tr_ds, list(range(min(args.limit, len(tr_ds)))))
        val_ds = Subset(val_ds, list(range(min(max(1, args.limit // 4), len(val_ds)))))

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        start = time.time()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if args.dry_run and n_batches >= args.dry_run_batches:
                break
        avg_loss = running_loss / max(1, n_batches)
        logger.info(
            "Epoch %d/%d train_loss=%.4f time=%.1fs",
            epoch,
            args.epochs,
            avg_loss,
            time.time() - start,
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_dice += dice_score(logits, yb)
                val_batches += 1
                if args.dry_run and val_batches >= max(1, args.dry_run_batches):
                    break
        val_loss /= max(1, val_batches)
        val_dice /= max(1, val_batches)
        logger.info("          val_loss=%.4f val_dice=%.4f", val_loss, val_dice)

        if args.dry_run:
            break

        if val_dice > best_val_dice + 1e-4:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                args.output,
            )
            logger.info("Saved best checkpoint to %s", args.output)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Train the lung segmenter.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=settings.segmenter_weights,
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

    train(args)


if __name__ == "__main__":
    main()
