import pytest

from src.detector.evaluate import aggregate_auc
from src.detector.model import BinaryMetrics


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
