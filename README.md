# README.md

A deep learning-based medical imaging platform that detects, segments, and
explains lung nodules in CT slices, served through a FastAPI backend with a
Next.js web frontend.

> **Disclaimer: This is a research/demo project, not a medical diagnostic
> device. Results must not be used for clinical decision-making.**

## What it does

Given a single CT slice (PNG/JPG), the system runs a three-stage inference
pipeline and returns a structured JSON payload:

1. **Classification** — is there a lung nodule? (`predicted_class`,
   `confidence`, per-class `probabilities`, and an `is_uncertain` flag).
2. **Segmentation** — a U-Net binary mask localizing the nodule, a bounding
   box, the nodule area in pixels, and a JET confidence map.
3. **Explainability** — a GradCAM heatmap and a blended overlay highlighting
   the regions the classifier found most important.

The full API contract is defined by the Pydantic models in
`backend/app/schemas.py`.

## Architecture

```
Lung Cancer Detection System/
├── backend/                 FastAPI inference service (Python)
│   ├── app/
│   │   ├── main.py          FastAPI app (/health, /predict)
│   │   ├── schemas.py       Response models (fixed API contract)
│   │   ├── inference.py     Orchestrates the inference pipeline
│   │   ├── config.py        Settings (read from env / backend/.env)
│   │   ├── models/          Classifier, segmenter, GradCAM
│   │   ├── utils/           Pre/postprocessing (PNG base64, masks, heatmaps)
│   │   ├── datasets/        Dataset loaders + label maps
│   │   └── weights/         Trained checkpoints (classifier.pth, segmenter.pth)
│   └── scripts/             Training + dataset build tools
├── frontend/                Next.js web UI (TypeScript + Tailwind)
├── notebooks/               Placeholder only — actual training ran on Colab
├── data/                    Dataset manifests and splits
└── Lung_Nodule_Dataset/     Raw Kaggle lung-nodule dataset
```

## Setup

### Prerequisites

- Python 3.9+ (with PyTorch, torchvision, OpenCV, FastAPI, uvicorn, Pillow)
- Node.js 18+ and npm

### 1. Backend

```bash
cd backend
pip install -r requirements.txt        # if requirements.txt is present
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Configuration can be overridden via `backend/.env` or environment variables —
see `backend/app/config.py` (`api_host`, `api_port`, `max_file_size_mb`,
`confidence_threshold`, `cors_origins`, `model_version`, etc.).

Trained weights are already shipped in `backend/app/weights/` (see
[Weights provenance](#weights-provenance)); with them present, `/health` returns
`"models_loaded": true`. If the weights are ever removed, the backend falls back
to **demo mode** — it responds successfully with random/pretrained-only
predictions so the full UI can still be exercised. See `TRAINING.md` to produce
new checkpoints.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend calls the backend at
`http://127.0.0.1:8000` — override with the `NEXT_PUBLIC_API_URL` environment
variable if the backend is hosted elsewhere.

For a production build:

```bash
npm run build
npm run start
```

## API endpoints

### `GET /health`

```json
{ "status": "healthy", "models_loaded": false, "gpu_available": false }
```

### `POST /predict`

Multipart form-data, field name `image`, containing a PNG or JPG file (max
10MB). Returns the full prediction payload described in `backend/app/schemas.py`:

```json
{
  "classification": {
    "predicted_class": "Healthy",
    "confidence": 0.5003,
    "is_uncertain": true,
    "probabilities": { "Healthy": 0.5003, "Lung_Nodule": 0.4997 }
  },
  "segmentation": {
    "has_nodule": true,
    "mask_base64": "data:image/png;base64,...",
    "bounding_box": { "x_min": 0.0, "y_min": 0.0, "x_max": 511.0, "y_max": 511.0 },
    "nodule_area_pixels": 262144,
    "confidence_map": "data:image/png;base64,..."
  },
  "gradcam": {
    "heatmap_base64": "data:image/png;base64,...",
    "overlay_base64": "data:image/png;base64,...",
    "explanation": "Heatmap shows regions most important for classification"
  },
  "processing_time_seconds": 4.366,
  "model_version": "v1.0.0",
  "device_used": "cpu"
}
```

All images are returned as `data:image/png;base64,...` Data URIs that the
frontend renders directly.

## Local background launch (Windows)

`run_server_hidden.vbs` launches uvicorn as a detached process that is **not**
subject to the shell's Windows Job-Object cleanup, so the server persists after
a terminal/tool call ends:

```bash
wscript.exe "E:\Temp\Lung Cancer Detection System\backend\run_server_hidden.vbs"
```

## Frontend features

- Drag-and-drop upload with client-side validation (PNG/JPG, 10MB max)
- Image preview with zoom/pan
- Classification result with per-class probability bars and an uncertainty badge
- Segmentation mask overlay with opacity + bounding box toggles
- GradCAM heatmap/overlay viewer with a transparency slider
- Animated transitions between upload → loading → results states

## Verification

With both services running, a full `/predict` round trip against a real scan:

```bash
curl.exe -F "image=@Lung_Nodule_Dataset/Lung_Nodule_Dataset/Images/1.png" \
  http://127.0.0.1:8000/predict | ConvertFrom-Json
```

## Weights provenance

The shipped checkpoints (`backend/app/weights/classifier.pth` and
`segmenter.pth`) come from a **Colab GPU training run** whose checkpoints were
saved to **Google Drive** and copied into this repository. They were **not**
produced by a notebook committed in this repo — `notebooks/` is a placeholder
only (see `TRAINING.md` and `notebooks/README.md`). If either file is missing
the backend falls back to demo mode and `/health` reports
`"models_loaded": false`.

## Known limitations

- **Proof-of-concept only:** the classifier shows **very high, uniform
  confidence (~100%) on its own training distribution**
  (`Lung_CT_Class_Dataset`). This most likely reflects strong separability
  between that dataset's two classes rather than proven real-world
  generalization; the same near-100% confidence persists even on unfamiliar
  data the model never trained on. These predictions must be treated as a
  research/demo output, **not a clinical claim**.
- **Research only:** the models are trained on limited public Kaggle data and
  are not validated for clinical use.
