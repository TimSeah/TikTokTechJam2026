from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detector.features import FeatureArrays

ARTIFACT_SCHEMA_VERSION = 1
FINAL_MODEL_NAME = "hybrid_augmented"
MODEL_NAMES = ("semantic_clean", "hybrid_clean", FINAL_MODEL_NAME)


def resolve_feature_mode(config: dict) -> str:
    mode = config.get("final_feature_mode", "hybrid")
    if mode not in {"semantic", "hybrid"}:
        raise ValueError(f"Unsupported final feature mode: {mode}")
    return mode


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    accuracy: float
    f1: float


def combine_features(arrays: FeatureArrays, semantic_only: bool) -> np.ndarray:
    semantic = arrays.semantic.astype(np.float32, copy=False)
    if semantic_only:
        return semantic
    return np.concatenate(
        (semantic, arrays.frequency.astype(np.float32, copy=False)), axis=1
    )


def _new_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs"),
            ),
        ]
    )


def validate_aligned(clean: FeatureArrays, augmented: FeatureArrays) -> None:
    if len(clean) != len(augmented):
        raise ValueError("Clean and augmented caches have different row counts")
    if not np.array_equal(clean.image_ids, augmented.image_ids):
        raise ValueError("Clean and augmented cache image IDs are not aligned")
    if not np.array_equal(clean.labels, augmented.labels):
        raise ValueError("Clean and augmented cache labels are not aligned")


def train_models(
    clean: FeatureArrays, augmented: FeatureArrays, seed: int
) -> dict[str, Pipeline]:
    validate_aligned(clean, augmented)
    semantic_clean = _new_pipeline(seed)
    hybrid_clean = _new_pipeline(seed)
    hybrid_augmented = _new_pipeline(seed)

    semantic_clean.fit(combine_features(clean, semantic_only=True), clean.labels)
    hybrid_clean.fit(combine_features(clean, semantic_only=False), clean.labels)
    hybrid_augmented.fit(
        np.concatenate(
            (
                combine_features(clean, semantic_only=False),
                combine_features(augmented, semantic_only=False),
            )
        ),
        np.concatenate((clean.labels, augmented.labels)),
    )
    return {
        "semantic_clean": semantic_clean,
        "hybrid_clean": hybrid_clean,
        FINAL_MODEL_NAME: hybrid_augmented,
    }


def predict_scores(
    model: Pipeline, arrays: FeatureArrays, semantic_only: bool
) -> np.ndarray:
    scores = model.predict_proba(combine_features(arrays, semantic_only))[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite predictions")
    return scores


def predict_margins(
    model: Pipeline, arrays: FeatureArrays, semantic_only: bool
) -> np.ndarray:
    margins = model.decision_function(combine_features(arrays, semantic_only))
    if not np.isfinite(margins).all():
        raise ValueError("Model produced non-finite decision margins")
    return np.asarray(margins, dtype=np.float64)


def calculate_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> BinaryMetrics:
    predictions = (scores >= threshold).astype(np.int8)
    return BinaryMetrics(
        auc=float(roc_auc_score(labels, scores)),
        accuracy=float(accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions)),
    )


def score_models(
    models: dict[str, Pipeline], arrays: FeatureArrays, threshold: float = 0.5
) -> dict[str, BinaryMetrics]:
    return {
        name: calculate_metrics(
            arrays.labels,
            predict_scores(model, arrays, semantic_only=name == "semantic_clean"),
            threshold,
        )
        for name, model in models.items()
    }


def save_artifact(artifact: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(output_path)


def load_artifact(artifact_path: Path) -> dict:
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
