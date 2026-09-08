# Lung Cancer Detection System

A deep-learning medical imaging platform that classifies, segments, and explains lung nodules in CT slices, served through a FastAPI backend with a Next.js web frontend.

**Pipeline:** ResNet-50 classification → DeepLabV3 segmentation → Grad-CAM heatmap visualization.

> **Disclaimer: This is a research/demo project, not a medical diagnostic device.
> Results must not be used for clinical decision-making.**

## Project Overview

Given a single CT slice (PNG/JPG), the system runs a three-stage inference pipeline:

1. **Classification** — ResNet-50 binary classifier: is there a lung nodule?
   Returns `predicted_class`, `confidence`, per-class `probabilities`, and an
   `is_uncertain` flag (when confidence < 0.70).
2. **Segmentation** — DeepLabV3-ResNet101 U-Net binary mask localizing the
   nodule, with a bounding box, pixel area, and a JET confidence map.
3. **Explainability** — Grad-CAM heatmap and blended overlay highlighting the
   regions the classifier found most important.

The full API contract is defined by the Pydantic models in `backend/app/schemas.py`.

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **Node.js 18+** and npm
- **Git LFS**

> **Critical note:** Cloning without Git LFS installed leaves the `.pth` model
> files as ~130-byte pointer stubs instead of real weights. Install Git LFS
> **before** cloning, and always run `git lfs pull` after cloning to download the
> actual weight files.
> - Windows: install from <https://git-lfs.github.com>
> - macOS: `brew install git-lfs`
> - Linux: `sudo apt install git-lfs` (or your distro's equivalent)

## Step-by-Step Setup Guide

### Step 1: Clone Repository & Pull Large Files

```bash
git clone <repo-url>
cd Lung-Cancer-Detection-System
git lfs pull   # download the real weight files
```

### Step 2: Download & Extract Dataset

1. Download the dataset from Kaggle:
   <https://www.kaggle.com/datasets/ucimachinelearning/lung-nodule-dataset>
2. Extract the downloaded folders directly into the project **ROOT** directory so
   the layout matches:

```
Lung-Cancer-Detection-System/
├── Lung_CT_Class_Dataset/
│   ├── Healthy/
│   └── Lung_Nodule/
└── Lung_Nodule_Dataset/
    ├── Images/
    └── Annotations/
```

> **Note:** The raw dataset folders live directly in the repository root (they are
> listed in `.gitignore` so the raw image files are never committed to Git).

### Step 3: Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows — on macOS/Linux use: source venv/bin/activate
pip install -r requirements.txt
```

Verify the weights exist (not Git LFS pointer stubs):

```bash
ls -lh backend/app/weights/classifier.pth   # ~99 MB
ls -lh backend/app/weights/segmenter.pth    # ~245 MB
```

Start the backend server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Frontend Setup

```bash
cd ../frontend
npm install
```

Create `frontend/.env.local` in the `frontend/` directory:

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the web interface:

```bash
npm run dev
```

### Step 5: Verify & Run Inference

1. Open <http://localhost:3000> in your browser.
2. Test the backend health endpoint:
   ```bash
   curl http://127.0.0.1:8000/health
   # → { "status": "healthy", "models_loaded": true, "gpu_available": false }
   ```
3. Upload a sample CT scan from `Lung_CT_Class_Dataset/` or `Lung_Nodule_Dataset/`
   to test inference, segmentation, and Grad-CAM generation.

If the weights loaded correctly, `/health` will return `"models_loaded": true`. If
either `.pth` file is missing or a stub, the backend falls back to **demo mode** —
the API still responds but with random/meaningless predictions, and `/health`
reports `"models_loaded": false`.

Configuration can be overridden via `backend/.env` or environment variables — see
`backend/app/config.py` for available keys (`api_host`, `api_port`,
`max_file_size_mb`, `confidence_threshold`, `cors_origins`, `model_version`, etc.).

## Project Structure

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

## Weights Provenance

The shipped checkpoints were produced by a Colab GPU training run and copied into
this repository. Previous versions are preserved in `backend/app/weights/versions/`:

| Version | Notes |
|---------|-------|
| `v2_regularized` | Label smoothing, Dice-weighted loss, group-aware splits |
| `v3` | Current active weights |

Use `backend/scripts/switch_weights.py <version>` to swap the active weights to a
saved version. See `TRAINING.md` for full training instructions and
hyperparameters.

## Known Limitations & Domain Shift Notes

> This is an **academic proof-of-concept**, not a clinical device.

- **In-distribution performance:** the classifier was trained on
  `Lung_CT_Class_Dataset` and `Lung_Nodule_Dataset` (public Kaggle data) and
  performs well in-distribution (~92–96% confidence, correctly varied across
  samples).
- **External dataset performance / domain shift:** when tested on a genuinely
  external dataset (IQ-OTH/NCCD), accuracy dropped significantly (~50%). This is
  attributed to a **brightness/windowing domain shift** between institutions —
  different CT scanner settings and windowing parameters produce images with
  substantially different intensity distributions.
- **Uniform confidence on unfamiliar data:** on its own training distribution, the
  classifier can produce very high, near-100% confidence predictions, reflecting
  strong separability between the two classes in that dataset rather than proven
  real-world generalization.
- **Limited training data:** both models were trained on small public Kaggle
  datasets, not large-scale clinical cohorts. No external validation was performed
  beyond the IQ-OTH/NCCD spot-check described above.
- **No clinical validation:** the models have not been validated against any
  clinical benchmark and must not be used for diagnostic decisions.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Service status, model load state, GPU availability |
| `POST` | `/predict` | Upload a CT slice → full prediction payload |

`POST /predict` accepts multipart form-data (field name `image`, PNG/JPG, max 10
MB) and returns the full prediction including classification, segmentation mask,
bounding box, Grad-CAM heatmap, and overlay — all as base64-encoded Data URIs.

## Frontend Features

- Drag-and-drop upload with client-side validation (PNG/JPG, 10 MB max)
- Image preview with zoom/pan
- Classification result with per-class probability bars and an uncertainty badge
- Segmentation mask overlay with opacity slider and bounding-box toggle
- Grad-CAM heatmap/overlay viewer with transparency slider
- Animated transitions between upload → loading → results states
