# Lung Cancer Detection System

A deep-learning medical imaging platform that classifies, segments, and explains lung nodules in CT slices, served through a FastAPI backend with a Next.js web frontend.

> **Disclaimer: This is a research/demo project, not a medical diagnostic device.
> Results must not be used for clinical decision-making.**

## What it does

Given a single CT slice (PNG/JPG), the system runs a three-stage inference pipeline:

1. **Classification** — ResNet-50 binary classifier: is there a lung nodule?
   Returns `predicted_class`, `confidence`, per-class `probabilities`, and an
   `is_uncertain` flag (when confidence < 0.70).
2. **Segmentation** — DeepLabV3-ResNet101 U-Net binary mask localizing the
   nodule, with a bounding box, pixel area, and a JET confidence map.
3. **Explainability** — GradCAM heatmap and blended overlay highlighting the
   regions the classifier found most important.

The full API contract is defined by the Pydantic models in
`backend/app/schemas.py`.

## Prerequisites

- **Python 3.9+** (tested with 3.13)
- **Node.js 18+** and npm
- **Git LFS** — this repository stores model weights (`.pth` files) via
  [Git LFS](https://git-lfs.github.com). If you clone without Git LFS
  installed, `classifier.pth` and `segmenter.pth` will be tiny pointer files
  (~130 bytes each) instead of real weights, and the backend will run in
  **demo mode** (random predictions). Install Git LFS **before** cloning:
  - Windows: download the installer from <https://git-lfs.github.com>
  - macOS: `brew install git-lfs`
  - Linux: `sudo apt install git-lfs` (or your distro's equivalent)

## Clone

```bash
git clone https://github.com/Mayank4712/Lung-Cancer-Detection-System.git
cd Lung-Cancer-Detection-System
git lfs pull   # download the real weight files (safe to run even if LFS was installed before cloning)
```

Verify the weights are real (not pointer stubs):

```bash
ls -lh backend/app/weights/classifier.pth   # ~99 MB, not ~130 bytes
ls -lh backend/app/weights/segmenter.pth    # ~245 MB, not ~130 bytes
```

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows — on macOS/Linux use: source .venv/bin/activate
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If the weights loaded correctly, `/health` will return `"models_loaded": true`:

```bash
curl http://127.0.0.1:8000/health
```

If either `.pth` file is missing or corrupted, the backend falls back to **demo
mode** — the API still responds but with random/pretrained-only (meaningless)
predictions. `/health` reports `"models_loaded": false` in that case.

Configuration can be overridden via `backend/.env` or environment variables —
see `backend/app/config.py` for all available keys (`api_host`, `api_port`,
`max_file_size_mb`, `confidence_threshold`, `cors_origins`, `model_version`,
etc.).

## Frontend setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local` (or copy from the example):

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the dev server:

```bash
npm run dev
```

Open <http://localhost:3000> in your browser.

For a production build:

```bash
npm run build
npm run start
```

## Verifying it works

1. With both services running, check the backend health:
   ```bash
   curl http://127.0.0.1:8000/health
   # → { "status": "healthy", "models_loaded": true, "gpu_available": false }
   ```
2. Open <http://localhost:3000> in your browser.
3. Upload a CT slice image (PNG or JPG, max 10 MB) via the drag-and-drop area.
   You should see classification results, a segmentation mask overlay, and a
   GradCAM heatmap.

You can also test via the CLI:

```bash
curl.exe -F "image=@Lung_Nodule_Dataset/Lung_Nodule_Dataset/Images/1.png" http://127.0.0.1:8000/predict
```

## Project structure

```
Lung Cancer Detection System/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app (/health, /predict)
│   │   ├── config.py            Settings (env / backend/.env)
│   │   ├── inference.py         Orchestrates the inference pipeline
│   │   ├── schemas.py           Response models (fixed API contract)
│   │   ├── models/              Classifier (ResNet-50), segmenter (DeepLabV3), GradCAM
│   │   ├── utils/               Pre/postprocessing (PNG base64, masks, heatmaps)
│   │   ├── datasets/            Dataset loaders + label maps
│   │   └── weights/             Trained checkpoints (Git LFS)
│   │       ├── classifier.pth   Active classifier weights
│   │       ├── segmenter.pth    Active segmenter weights
│   │       └── versions/        Saved weight snapshots (v2_regularized, v3)
│   ├── scripts/                 Training + dataset build tools
│   └── requirements.txt         Python dependencies
├── frontend/                    Next.js web UI (TypeScript + Tailwind)
│   ├── app/                     App Router pages + components
│   ├── lib/                     API client, Zustand store, TS types
│   └── package.json
├── notebooks/                   Placeholder — actual training ran on Colab
├── data/                        Dataset manifests and splits
├── Lung_CT_Class_Dataset/       Raw Kaggle classification dataset (gitignored)
├── Lung_Nodule_Dataset/         Raw Kaggle segmentation dataset (gitignored)
├── TRAINING.md                  How to retrain the models from scratch
└── README.md
```

## Weights provenance

The shipped checkpoints were produced by a Colab GPU training run and copied
into this repository. Previous versions are preserved in
`backend/app/weights/versions/`:

| Version | Notes |
|---------|-------|
| `v2_regularized` | Label smoothing, Dice-weighted loss, group-aware splits |
| `v3` | Current active weights |

Use `backend/scripts/switch_weights.py <version>` to swap the active weights to
a saved version. See `TRAINING.md` for full training instructions and
hyperparameters.

## Known limitations

- **Proof-of-concept only:** the classifier was trained on `Lung_CT_Class_Dataset`
  and `Lung_Nodule_Dataset` (public Kaggle data). It performs well in-distribution
  (~92–96% confidence, correctly varied across samples) but showed a significant
  accuracy drop (~50%) when tested on a genuinely external dataset
  (IQ-OTH/NCCD). This is attributed to a brightness/windowing domain shift
  between institutions — different CT scanner settings and windowing
  parameters produce images with substantially different intensity
  distributions.
- **Uniform confidence on unfamiliar data:** on its own training distribution,
  the classifier can produce very high, near-100% confidence predictions. This
  most likely reflects strong separability between that dataset's two classes
  rather than proven real-world generalization.
- **Limited training data:** both models were trained on small public Kaggle
  datasets, not large-scale clinical cohorts. No external validation was
  performed beyond the IQ-OTH/NCCD spot-check described above.
- **No clinical validation:** the models have not been validated against any
  clinical benchmark and must not be used for diagnostic decisions.

> **Bottom line:** this is a research/proof-of-concept tool demonstrating a
> classification + segmentation + explainability pipeline on lung CT scans. It
> is **not** a clinical device.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Service status, model load state, GPU availability |
| `POST` | `/predict` | Upload a CT slice → full prediction payload |

`POST /predict` accepts multipart form-data (field name `image`, PNG/JPG, max 10
MB) and returns the full prediction including classification, segmentation mask,
bounding box, GradCAM heatmap, and overlay — all as base64-encoded Data URIs.

## Frontend features

- Drag-and-drop upload with client-side validation (PNG/JPG, 10 MB max)
- Image preview with zoom/pan
- Classification result with per-class probability bars and an uncertainty badge
- Segmentation mask overlay with opacity slider and bounding-box toggle
- GradCAM heatmap/overlay viewer with transparency slider
- Animated transitions between upload → loading → results states
