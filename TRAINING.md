# TRAINING.md

How to produce the trained checkpoints that the inference service loads, and how
to place them where the backend expects them.

## Overview

The backend loads two PyTorch checkpoints at startup:

- `backend/app/weights/classifier.pth` — binary lung-nodule classifier
- `backend/app/weights/segmenter.pth` — U-Net segmentation model

If either file is missing, the backend logs a clear warning and runs in **demo
mode** (random/pretrained-only backbone), so `/predict` still responds with an
API-valid but meaningless prediction. `/health` reports `"models_loaded": false`
until **both** real checkpoints are present.

> **How the shipped weights were produced:** the `classifier.pth` and
> `segmenter.pth` in this repo came from a **Colab GPU training run**, with
> checkpoints saved to **Google Drive** and later copied into
> `backend/app/weights/`. They were **not** trained by a notebook committed in
> this repository — `notebooks/` is a placeholder only. The scripts referenced
> below remain the canonical way to reproduce training (or start fresh runs).

## Where training should run

Training is GPU-heavy and designed to run on **Colab or Kaggle GPU** (a local
environment typically has no CUDA, per this project's ground rules). The training
scripts themselves live in `backend/scripts/` and are written as clean, runnable
CLI modules that are safe to smoke-test on CPU.

Data is expected under the repository and its paths are resolved from
`backend/app/config.py` (they are not passed as CLI flags):

- Classification: `Lung_CT_Class_Dataset/` (Healthy / Lung_Nodule directories),
  with the manifest and splits in `data/labels.csv` and `data/splits.json`.
- Segmentation: `Lung_Nodule_Dataset/Lung_Nodule_Dataset/` with `Images/` and
  `Annotations/`.

## Training the classifier

```bash
cd backend
python scripts/train_classifier.py --output app/weights/classifier.pth
```

Fixed hyperparameters: Adam lr=1e-3, batch size 16, 10 epochs,
class-weighted CrossEntropy, ReduceLROnPlateau on validation loss. The best
checkpoint (by validation loss) is written to `--output`, which defaults to
`app/weights/classifier.pth`.

## Training the segmenter

```bash
cd backend
python scripts/train_segmenter.py --output app/weights/segmenter.pth
```

Fixed hyperparameters: Adam lr=1e-4, batch size 8 (4 fallback), loss
`0.5 * BCEWithLogits + 0.5 * Dice`, grad clip 1.0, early stop (patience 10) on
validation Dice. The best checkpoint is written to `--output`, which defaults to
`app/weights/segmenter.pth`.

## Optional: notebooks

Heavyweight Colab/Kaggle notebooks are intentionally not committed to this
repository (see `notebooks/README.md`). A notebook is simply an interactive
wrapper around the two scripts above — running the same training loop on GPU
and exporting the resulting state dicts to `backend/app/weights/`.

## Producing a `.pth` file

Both scripts save the **model state dict** (optionally wrapped as
`{"model": ...}` — the inference service handles both forms via
`inference.py::_state_dict_from_ckpt`). If you train in a notebook:

1. Run the classifier/segmenter training loop (directly, or by importing the
   scripts).
2. Save the weights so the keys match the model architectures in
   `backend/app/models/`.
3. Upload `classifier.pth` and `segmenter.pth` to `backend/app/weights/`.
4. Restart the backend. `/health` should now report `"models_loaded": true`.

## Verifying weights are loaded

```bash
curl.exe http://127.0.0.1:8000/health
# { "status": "healthy", "models_loaded": true, "gpu_available": false }
```

## Local CPU smoke test (optional)

The training scripts support a `--dry-run` that exercises the full
data-loader → model → loss → backward path on a tiny CPU subset without saving
weights:

```bash
cd backend
python scripts/train_classifier.py --dry-run --epochs 1 --limit 32
python scripts/train_segmenter.py --dry-run --epochs 1 --limit 32
```
