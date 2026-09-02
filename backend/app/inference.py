"""LungCancerInferenceService.

Orchestrates the full inference pipeline for a single uploaded image:

    preprocessing -> classifier forward -> segmenter forward -> GradCAM ->
    postprocessing

Design ("demo mode"): on init it attempts to load app/weights/classifier.pth and
app/weights/segmenter.pth. If either checkpoint is missing (e.g. before any
Colab training has produced .pth files) it logs a clear warning and keeps a
randomly-initialized / pretrained-only backbone so the API still responds
successfully with (admittedly meaningless) predictions. `models_loaded` in
/health is True only when BOTH real checkpoints were loaded.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.models.classifier import LungClassifier
from app.models.gradcam import GradCAM
from app.models.segmenter import LungSegmenter
from app.utils import postprocessing, preprocessing

logger = logging.getLogger(__name__)

CLASSIFIER_WEIGHTS = Path(settings.classifier_weights)
SEGMENTER_WEIGHTS = Path(settings.segmenter_weights)

UNCERTAINTY_EXPLANATION = (
    "Heatmap shows regions most important for classification"
)

# Classifier-segmenter agreement gate: a bounding box is only reported when the
# classifier predicts a nodule with at least this confidence AND the segmenter
# detects a connected region of at least this many pixels (small blobs are mostly
# noise on healthy scans, but genuinely small nodules should not be discarded).
BOUNDING_BOX_MIN_CONFIDENCE = 0.60
BOUNDING_BOX_MIN_AREA = 15

# Segmentation sigmoid threshold is now configurable via the SEG_THRESHOLD
# environment variable (default 0.5).  See config.py for the setting.

# When the classifier is strongly confident of a nodule but the segmenter found
# nothing valid, fall back to the GradCAM heatmap to localize the suspicious
# area. Only used at or above this confidence.
GRADCAM_FALLBACK_MIN_CONFIDENCE = 0.70
# The fallback heatmap is thresholded at this fraction of its peak to isolate the
# dominant activation region.
GRADCAM_FALLBACK_PEAK_FRACTION = 0.60
# Fallback bounding boxes smaller than this area (in displayed-image pixels) are
# degenerate noise slivers (e.g. a 5x1 px residual) rather than a localized
# suspicious region, so they are rejected. Genuinely small nodules, like the
# 24 px^2 one, are comfortably above this and are kept.
GRADCAM_FALLBACK_MIN_BOX_AREA = 15


def _state_dict_from_ckpt(path: Path, device: torch.device):
    """Load a checkpoint: handle a bare state_dict or a {'model': ...} wrapper."""
    ckpt = torch.load(path, map_location=str(device))
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def _remap_classifier_state(state: dict) -> dict:
    """Remap Colab checkpoint keys: head.X -> backbone.fc.X."""
    remapped = {}
    for k, v in state.items():
        if k.startswith("head."):
            remapped["backbone.fc." + k[len("head."):]] = v
        else:
            remapped[k] = v
    return remapped


def _remap_segmenter_state(state: dict) -> dict:
    """Remap Colab checkpoint keys to match LungSegmenter's expected layout.

    The Colab checkpoint was saved from a raw deeplabv3_resnet101 so keys are:
      model.backbone.*, model.classifier.*, model.aux_classifier.*
    The LungSegmenter class registers self.backbone = self.model.backbone,
    which means state_dict expects both 'backbone.*' and 'model.backbone.*'.
    We duplicate model.backbone.* keys as backbone.* and drop the model.backbone
    prefix (keeping both sets so the shared-parameter structure loads cleanly).
    """
    remapped = {}
    for k, v in state.items():
        if k.startswith("model.aux_classifier."):
            # Auxiliary head is only used during training (aux loss); it is not
            # consumed for the "out" segmentation we return. Skip it so loading
            # works even when the model has no aux_head (use_pretrained=False).
            continue
        if k.startswith("model.backbone."):
            suffix = k[len("model."):]  # backbone.X
            remapped[k] = v           # model.backbone.X
            remapped[suffix] = v      # backbone.X (duplicate for self.backbone)
        else:
            remapped[k] = v
    return remapped


class LungCancerInferenceService:
    def __init__(self) -> None:
        self.device = torch.device(settings.device)

        # Trained segmentation/classifier weights can produce tiny (denormal)
        # activations on normalized inputs, which collapse CPU conv throughput
        # (~50x). Flush-to-zero avoids the slow denormal path; harmless to
        # inference accuracy. Ignored where unsupported (e.g. GPU).
        try:
            torch.set_flush_denormal(True)
        except Exception:  # pragma: no cover - platform without FTZ support
            pass

        self.classifier = LungClassifier(use_pretrained=settings.use_pretrained)
        self.segmenter = LungSegmenter(use_pretrained=settings.use_pretrained)

        self.has_classifier_weights = False
        self.has_segmenter_weights = False

        self._load_checkpoints()

        self.classifier.to(self.device).eval()
        self.segmenter.to(self.device).eval()

        self.gradcam_clf = GradCAM(
            self.classifier, self.classifier.target_layer()
        )
        self.gradcam_seg = GradCAM(
            self.segmenter, self.segmenter.target_layer()
        )

        logger.info(
            "Inference service ready | device=%s | classifier_weights=%s "
            "segmenter_weights=%s",
            self.device,
            self.has_classifier_weights,
            self.has_segmenter_weights,
        )

    def _load_checkpoints(self) -> None:
        if CLASSIFIER_WEIGHTS.exists():
            try:
                state = _state_dict_from_ckpt(CLASSIFIER_WEIGHTS, self.device)
                state = _remap_classifier_state(state)
                self.classifier.load_state_dict(state)
                self.has_classifier_weights = True
                logger.info("Loaded classifier weights from %s", CLASSIFIER_WEIGHTS)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load classifier weights (%s): %s",
                    CLASSIFIER_WEIGHTS,
                    exc,
                )
        else:
            logger.warning(
                "Classifier weights not found at %s — running in demo mode with "
                "random/pretrained-only backbone (predictions are meaningless).",
                CLASSIFIER_WEIGHTS,
            )

        if SEGMENTER_WEIGHTS.exists():
            try:
                state = _state_dict_from_ckpt(SEGMENTER_WEIGHTS, self.device)
                state = _remap_segmenter_state(state)
                self.segmenter.load_state_dict(state)
                self.has_segmenter_weights = True
                logger.info("Loaded segmenter weights from %s", SEGMENTER_WEIGHTS)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load segmenter weights (%s): %s",
                    SEGMENTER_WEIGHTS,
                    exc,
                )
        else:
            logger.warning(
                "Segmenter weights not found at %s — running in demo mode with "
                "random/pretrained-only backbone (predictions are meaningless).",
                SEGMENTER_WEIGHTS,
            )

    @property
    def models_loaded(self) -> bool:
        return self.has_classifier_weights and self.has_segmenter_weights

    @property
    def gpu_available(self) -> bool:
        return torch.cuda.is_available()

    def predict(self, image: Image.Image) -> dict:
        """Run the full pipeline on a PIL image and return the API response dict."""
        start = time.time()

        clf_input = preprocessing.preprocess_classifier(image)
        clf_tensor = (
            torch.from_numpy(clf_input).unsqueeze(0).to(self.device)
        )  # (1, 1, 256, 256)

        seg_input = preprocessing.preprocess_segmentation(image)
        seg_tensor = (
            torch.from_numpy(seg_input).unsqueeze(0).to(self.device)
        )  # (1, 1, 512, 512)

        # --- Classifier ---
        with torch.no_grad():
            clf_logits = self.classifier(clf_tensor)
        probs = torch.softmax(clf_logits[0], dim=0).detach().cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = "Lung_Nodule" if pred_idx == 1 else "Healthy"
        confidence = float(probs[pred_idx])
        probabilities = {
            "Healthy": float(probs[0]),
            "Lung_Nodule": float(probs[1]),
        }
        is_uncertain = confidence < settings.confidence_threshold

        # --- Segmenter ---
        with torch.no_grad():
            seg_logits = self.segmenter(seg_tensor)
        seg_np = seg_logits[0].detach().cpu().numpy()  # (1, 512, 512)
        seg_result = postprocessing.process_segmentation(
            seg_np, confidence_threshold=settings.seg_threshold
        )

        # --- GradCAM on the classifier (needs to run before fallback) ---
        original_rgb = np.asarray(image.convert("RGB"))
        clf_cam = self.gradcam_clf.generate(clf_tensor)
        # Zero edge activations from the border shortcut using the same ROI mask
        # applied to the classifier input (consistent, class-agnostic geometry).
        cam_mask = preprocessing.central_roi_mask(clf_cam.shape[:2])
        clf_cam = clf_cam * cam_mask
        # CAM is 256x256; resize to original for a faithful overlay.
        clf_cam_resized = _resize_cam(clf_cam, original_rgb)

        # --- Classifier-segmenter agreement gating for bounding boxes ---
        # A box is only trusted when the classifier calls it a nodule with
        # sufficient confidence AND the segmenter region is large enough to be a
        # real nodule rather than a noise blob. Otherwise detection_boxes is
        # strictly empty and has_nodule is False.
        classifier_says_nodule = pred_class == "Lung_Nodule" and (
            confidence >= BOUNDING_BOX_MIN_CONFIDENCE
        )
        seg_area = int(seg_result["nodule_area_pixels"])
        seg_bbox = seg_result["bounding_box"]
        valid_bbox = (
            seg_bbox["x_min"] is not None
            and seg_area >= BOUNDING_BOX_MIN_AREA
        )

        detection_boxes: list[dict] = []
        if classifier_says_nodule and valid_bbox:
            # Map the segmenter bbox (computed in 512x512 mask space) into the
            # original-image coordinate space so it renders on top of the
            # displayed CT slice consistently with the GradCAM fallback box.
            detection_boxes.append(
                _seg_bbox_to_original(seg_bbox, original_rgb.shape[:2])
            )

        # --- GradCAM fallback bounding box ---
        # If the classifier is strongly confident of a nodule but the segmenter
        # produced no valid box (threshold too strict, tiny nodule, or a nodule
        # sitting in the segmenter's blind spot), localize it from the GradCAM
        # heatmap so a bounding box is still displayed.
        if (
            not detection_boxes
            and pred_class == "Lung_Nodule"
            and confidence >= GRADCAM_FALLBACK_MIN_CONFIDENCE
        ):
            fallback = _gradcam_contour_bbox(clf_cam_resized)
            if fallback is not None:
                detection_boxes.append(fallback)

        nodule_detected = bool(detection_boxes)
        seg_result["has_nodule"] = nodule_detected
        seg_result["detection_boxes"] = detection_boxes
        if nodule_detected:
            # Keep the singular bounding_box field in the same original-image
            # coordinate space as detection_boxes.
            seg_result["bounding_box"] = detection_boxes[0]
        else:
            seg_result["bounding_box"] = {
                "x_min": None,
                "y_min": None,
                "x_max": None,
                "y_max": None,
            }

        gradcam_result = postprocessing.process_gradcam(
            clf_cam_resized,
            original_rgb,
            alpha=0.4,
        )
        gradcam_result["explanation"] = UNCERTAINTY_EXPLANATION

        processing_time = time.time() - start

        return {
            "classification": {
                "predicted_class": pred_class,
                "confidence": round(confidence, 4),
                "is_uncertain": is_uncertain,
                "probabilities": {
                    "Healthy": round(probabilities["Healthy"], 4),
                    "Lung_Nodule": round(probabilities["Lung_Nodule"], 4),
                },
            },
            "segmentation": seg_result,
            "gradcam": gradcam_result,
            "processing_time_seconds": round(processing_time, 3),
            "model_version": settings.model_version,
            "device_used": str(self.device),
        }


def _resize_cam(cam: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    import cv2

    if cam.shape[:2] == target_rgb.shape[:2]:
        return cam
    return cv2.resize(
        cam,
        (target_rgb.shape[1], target_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _gradcam_contour_bbox(cam: np.ndarray) -> dict | None:
    """Localize the dominant GradCAM activation as a bounding box.

    ``cam`` is a [0, 1] heatmap in displayed (original-image) coordinates that
    has already had its outer border neutralized. The heatmap is thresholded at
    GRADCAM_FALLBACK_PEAK_FRACTION of its peak and the largest connected contour
    is converted to a [x_min, y_min, x_max, y_max] bounding rectangle.

    Returns None when there is no meaningful activation to box.
    """
    import cv2

    cam = np.asarray(cam, dtype=np.float32)
    if cam.size == 0 or float(cam.max()) <= 0.0:
        return None

    binary = (cam > GRADCAM_FALLBACK_PEAK_FRACTION * float(cam.max())).astype(
        np.uint8
    ) * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 1 or h < 1:
        return None
    if (w * h) < GRADCAM_FALLBACK_MIN_BOX_AREA:
        # Degenerate noise sliver — not a localized suspicious region.
        return None

    return {
        "x_min": int(x),
        "y_min": int(y),
        "x_max": int(x + w),
        "y_max": int(y + h),
    }


def _seg_bbox_to_original(bbox: dict, original_shape: tuple[int, int]) -> dict:
    """Map a segmenter bounding box from 512x512 mask space to the original image.

    The segmenter center-crops the image to a square then resizes to 512x512, so
    its mask coordinates live in 512x512 space centered on the original. This
    reverses that transform so the box overlays the displayed CT slice correctly:
        orig_x = crop_origin_x + (bbox_512_x * crop_size / 512)
    where crop_size = min(orig_w, orig_h) and the crop is centered.
    """
    orig_h, orig_w = original_shape[:2]
    crop = float(min(orig_w, orig_h))
    scale = crop / 512.0
    ox = (orig_w - crop) / 2.0
    oy = (orig_h - crop) / 2.0
    return {
        "x_min": int(round(ox + bbox["x_min"] * scale)),
        "y_min": int(round(oy + bbox["y_min"] * scale)),
        "x_max": int(round(ox + bbox["x_max"] * scale)),
        "y_max": int(round(oy + bbox["y_max"] * scale)),
    }
