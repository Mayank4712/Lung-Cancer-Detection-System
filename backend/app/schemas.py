"""Pydantic response models for the /predict and /health endpoints.

These mirror the fixed API contract (section 5 of the spec).

Omission policy (documented): when the predicted segmentation mask is empty and
no bounding box can be found, `segmentation.bounding_box` is not omitted but its
four coordinates (x_min, y_min, x_max, y_max) are all set to None, and
`has_nodule` is False. This keeps the JSON shape stable for the frontend while
conveying "no region found" deterministically.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None


class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    is_uncertain: bool
    probabilities: dict[str, float]


class SegmentationResult(BaseModel):
    has_nodule: bool
    mask_base64: str
    bounding_box: BoundingBox
    nodule_area_pixels: int
    confidence_map: str


class GradCAMResult(BaseModel):
    heatmap_base64: str
    overlay_base64: str
    explanation: str


class PredictResponse(BaseModel):
    classification: ClassificationResult
    segmentation: SegmentationResult
    gradcam: GradCAMResult
    processing_time_seconds: float
    model_version: str
    device_used: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gpu_available: bool
