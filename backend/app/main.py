"""FastAPI application entrypoint.

Endpoints:
    GET  /health   -> {"status", "models_loaded", "gpu_available"}
    POST /predict  -> multipart image -> full prediction payload

Run (dev):  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import logging

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.inference import LungCancerInferenceService
from app.schemas import HealthResponse, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EXTS = (".png", ".jpg", ".jpeg")

app = FastAPI(title="Lung Cancer Detection API", version=settings.model_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the (lazily-constructed) inference service once at startup.
inference_service: LungCancerInferenceService | None = None


@app.on_event("startup")
def _startup() -> None:
    global inference_service
    if inference_service is None:
        inference_service = LungCancerInferenceService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = _require_service()
    return HealthResponse(
        status="healthy",
        models_loaded=service.models_loaded,
        gpu_available=service.gpu_available,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(..., description="PNG/JPG image")) -> dict:
    service = _require_service()

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a PNG or JPG image.",
        )
    name = (image.filename or "").lower()
    if not name.endswith(ALLOWED_EXTS):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file extension. Please upload a PNG or JPG image.",
        )

    data = await image.read()
    if len(data) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb}MB limit.",
        )

    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Could not read image. File may be corrupt."
        ) from exc

    try:
        return service.predict(pil)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def _require_service() -> LungCancerInferenceService:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return inference_service


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
