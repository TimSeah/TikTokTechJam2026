import numpy as np

from src.detector.shift_audit import (
    build_near_hash_index,
    near_hash_candidates,
    metadata_baseline,
    paired_bootstrap_auc_difference,
    reliability_diagnostics,
    source_score_summary,
    stratified_bootstrap_auc,
    threshold_diagnostics,
)
from tests.test_model import _arrays


def test_bootstrap_auc_and_paired_difference_are_deterministic() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    perfect = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    reversed_scores = perfect[::-1]

    result = stratified_bootstrap_auc(labels, perfect, replicates=20, seed=7)
    difference = paired_bootstrap_auc_difference(
        labels, perfect, reversed_scores, replicates=20, seed=7
    )

    assert result["auc"] == 1.0
    assert result["bootstrap_95_ci"] == [1.0, 1.0]
    assert difference["difference"] == 1.0
    assert difference["bootstrap_95_ci"] == [1.0, 1.0]


def test_threshold_and_calibration_diagnostics() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])

    threshold = threshold_diagnostics(labels, scores, threshold=0.5)
    calibration = reliability_diagnostics(labels, scores, bins=2)

    assert threshold["confusion_matrix_tn_fp_fn_tp"] == [2, 0, 0, 2]
    assert threshold["sensitivity"] == 1.0
    assert threshold["specificity"] == 1.0
    assert calibration["brier_score"] == 0.08500000000000002
    assert calibration["expected_calibration_error_10_bin"] == 0.25


def test_metadata_baseline_uses_out_of_fold_predictions() -> None:
    rows = []
    for label in (0, 1):
        for index in range(10):
            rows.append(
                {
                    "actual_width": str(100 + label * 100),
                    "actual_height": str(100 + label * 100),
                    "bytes": str(1000 + label * 1000 + index),
                    "label": str(label),
                }
            )

    result = metadata_baseline(rows, seeds=2)

    assert result["mean_auc"] == 1.0
    assert result["aucs"] == [1.0, 1.0]


def test_source_score_summary_marks_single_class_auc_undefined() -> None:
    arrays = _arrays()
    rows = [
        {"relative_path": path, "source": "real" if index < 2 else "fake"}
        for index, path in enumerate(arrays.relative_paths)
    ]

    result = source_score_summary(
        rows, arrays, np.asarray([0.1, 0.2, 0.8, 0.9]), threshold=0.5
    )

    assert result["real"]["predicted_fake_rate"] == 0.0
    assert result["fake"]["predicted_fake_rate"] == 1.0
    assert result["real"]["auc"] is None


def test_near_hash_index_finds_every_value_with_four_changed_bits() -> None:
    original = 0x123456789ABCDEF0
    changed = original ^ 0b1000100010001
    index = build_near_hash_index({original: {0}})

    candidates = near_hash_candidates(index, changed)

    assert original in candidates
    assert (original ^ changed).bit_count() == 4