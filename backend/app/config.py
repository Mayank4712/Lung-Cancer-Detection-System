"""Application settings (loaded from environment / .env).

Uses Pydantic Settings BaseSettings so values can be overridden via environment
variables or a backend/.env file. device auto-detects CUDA and falls back to CPU.
"""
from __future__ import annotations

import os
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - fallback for minimal envs
    from pydantic import BaseSettings

    _HAS_PYDANTIC_SETTINGS = False

    SettingsConfigDict = None


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """Central configuration.

    classifier_weights / segmenter_weights:
        Paths to trained checkpoints. If absent, the inference service falls back
        to randomly-initialized (or pretrained-only) backbones so the API still
        works in "demo mode".
    use_pretrained:
        Whether to load ImageNet/COCO pretrained backbones on init.
    device:
        Auto-detected cuda->cpu.
    use_mixed_precision:
        Enables GradScaler/autocast during training (GPU only).
    confidence_threshold:
        Below this the classifier reports is_uncertain=True.
    """

    classifier_weights: str = os.path.join(BASE_DIR, "app", "weights", "classifier.pth")
    segmenter_weights: str = os.path.join(BASE_DIR, "app", "weights", "segmenter.pth")
    use_pretrained: bool = False
    device: str = _auto_device()
    use_mixed_precision: bool = False
    confidence_threshold: float = 0.70
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_file_size_mb: int = 10
    inference_timeout_sec: int = 30

    model_version: str = "v1.0.0"

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=os.path.join(BASE_DIR, ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
