import numpy as np
import pytest

from src.detector.evaluate import aggregate_auc, evaluate_models
from src.detector.features import FeatureArrays
from src.detector.model import BinaryMetrics


class ShapeCheckingModel:
    def __init__(self, expected_features: int) -> None:
        self.expected_features = expected_features

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[1] == self.expected_features
        scores = np.asarray([0.25, 0.75], dtype=np.float32)
        return np.column_stack((1.0 - scores, scores))


def _metrics(auc: float) -> BinaryMetrics:
    return BinaryMetrics(auc=auc, accuracy=auc, f1=auc)


def test_aggregate_auc_distinguishes_condition_and_family_weighting() -> None:
    results = {
        "clean": _metrics(0.9),
        "jpeg_q90": _metrics(0.8),
        "jpeg_q50": _metrics(0.6),
        "blur_sigma1.0": _metrics(0.4),
    }
    summary = aggregate_auc(results)
    assert summary["condition_weighted_robust_auc"] == pytest.approx(0.6)
    assert summary["family_balanced_robust_auc"] == pytest.approx(0.55)
    assert summary["condition_weighted_final_score"] == pytest.approx(0.75)
    assert summary["family_balanced_final_score"] == pytest.approx(0.725)
    assert summary["family_auc"] == pytest.approx({"blur": 0.4, "jpeg": 0.7})


def test_evaluate_models_uses_declared_semantic_features() -> None:
    arrays = FeatureArrays(
        semantic=np.zeros((2, 2), dtype=np.float16),
        frequency=np.zeros((2, 1), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int8),
        image_ids=np.asarray(["real", "fake"]),
        relative_paths=np.asarray(["real.jpg", "fake.jpg"]),
        transform_keys=np.asarray(["clean", "clean"]),
    )

    metrics = evaluate_models(
        {"native": ShapeCheckingModel(expected_features=2)},
        arrays,
        threshold=0.5,
        semantic_models={"native"},
    )

    assert metrics["native"] == BinaryMetrics(auc=1.0, accuracy=1.0, f1=1.0)
