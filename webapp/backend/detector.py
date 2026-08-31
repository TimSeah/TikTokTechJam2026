from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps

from .config import Settings


ARTIFACT_SCHEMA_VERSION = 1


def _load_artifact(artifact_path: Path) -> dict:
    artifact = joblib.load(artifact_path)
    required = {"schema_version", "final_model", "models", "config"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing keys: {sorted(missing)}")
    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema: {artifact['schema_version']}")
    if artifact["final_model"] not in artifact["models"]:
        raise ValueError("Artifact final model is not present in model bundle")
    return artifact


def _resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested, but no GPU is available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _radial_fft_features(
    image: Image.Image, bins: int = 32, image_size: int = 256
) -> np.ndarray:
    if bins < 1 or image_size < 2:
        raise ValueError("bins must be positive and image_size must be at least 2")

    grayscale = ImageOps.grayscale(image).resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    values = np.asarray(grayscale, dtype=np.float32) / 255.0
    values -= values.mean()
    window = np.outer(np.hanning(image_size), np.hanning(image_size)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(values * window))))

    coordinates = np.arange(image_size, dtype=np.float32) - (image_size - 1) / 2
    y_grid, x_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(x_grid**2 + y_grid**2)
    radius /= radius.max()

    edges = np.linspace(0.0, 1.0, bins + 1)
    features = np.empty(bins, dtype=np.float32)
    for index in range(bins):
        upper_bound = radius <= edges[index + 1] if index == bins - 1 else radius < edges[index + 1]
        mask = (radius >= edges[index]) & upper_bound
        features[index] = float(spectrum[mask].mean()) if mask.any() else 0.0
    return features


@dataclass(frozen=True)
class Prediction:
    fake_probability: float
    label: str
    elapsed_ms: float


class DetectorService:
    def __init__(self, settings: Settings) -> None:
        if not settings.model_artifact_path.is_file():
            raise FileNotFoundError(
                f"Model artifact does not exist: {settings.model_artifact_path}"
            )

        artifact = _load_artifact(settings.model_artifact_path)
        self.config = artifact["config"]
        self.device = _resolve_device(settings.model_device)
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            str(self.config["model_name"]),
            pretrained=str(self.config["pretrained_checkpoint"]),
        )
        self.backbone = self.backbone.eval().to(self.device)
        self.classifier = artifact["models"][artifact["final_model"]]
        self.threshold = float(self.config["threshold"])
        self.display_name = f"{self.config['model_name']} / {artifact['final_model']}"
        self._inference_lock = threading.Lock()

    def predict(self, image_path: Path) -> Prediction:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        with self._inference_lock:
            started_at = time.perf_counter()
            semantic_input = self.preprocess(image).unsqueeze(0).to(self.device)
            frequency = _radial_fft_features(
                image,
                bins=int(self.config["fft_bins"]),
                image_size=int(self.config["fft_image_size"]),
            )

            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        semantic = self.backbone.encode_image(semantic_input)
                else:
                    semantic = self.backbone.encode_image(semantic_input)
                semantic = functional.normalize(semantic.float(), dim=-1)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            semantic_features = (
                semantic.cpu().numpy().astype(np.float16).astype(np.float32)
            )
            combined_features = np.concatenate(
                (semantic_features, frequency.reshape(1, -1)), axis=1
            )
            expected_dimension = int(self.config["semantic_dimension"]) + int(
                self.config["frequency_dimension"]
            )
            actual_dimension = combined_features.shape[1]
            if actual_dimension != expected_dimension:
                raise ValueError(
                    "Feature dimension mismatch: "
                    f"expected {expected_dimension}, got {actual_dimension}"
                )

            probabilities = self.classifier.predict_proba(combined_features)[:, 1]
            if not np.isfinite(probabilities).all():
                raise ValueError("Model produced non-finite predictions")
            fake_probability = float(probabilities[0])
            elapsed_ms = (time.perf_counter() - started_at) * 1000

        return Prediction(
            fake_probability=fake_probability,
            label="FAKE" if fake_probability >= self.threshold else "REAL",
            elapsed_ms=elapsed_ms,
        )
