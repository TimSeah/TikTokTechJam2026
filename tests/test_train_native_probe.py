import numpy as np

from src.detector.features import FeatureArrays
from src.detector.train_native_probe import (
    balanced_indices,
    calibrate_threshold,
    evaluate_gate,
    train_native_semantic_probe,
    validate_disjoint,
)


def _arrays(labels: list[int], offset: float) -> FeatureArrays:
    labels_array = np.asarray(labels, dtype=np.int8)
    row_count = len(labels)
    semantic = np.column_stack(
        (
            labels_array.astype(np.float32) + offset,
            np.arange(row_count, dtype=np.float32) / max(row_count, 1),
        )
    ).astype(np.float16)
    return FeatureArrays(
        semantic=semantic,
        frequency=np.zeros((row_count, 1), dtype=np.float32),
        labels=labels_array,
        image_ids=np.asarray([f"id-{offset}-{index}" for index in range(row_count)]),
        relative_paths=np.asarray([f"image-{index}.jpg" for index in range(row_count)]),
        transform_keys=np.asarray(["clean"] * row_count),
    )


def test_balanced_indices_are_deterministic() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)

    first = balanced_indices(labels, per_class=2, seed=7)
    second = balanced_indices(labels, per_class=2, seed=7)

    assert np.array_equal(first, second)
    assert np.array_equal(np.bincount(labels[first]), np.asarray([2, 2]))


def test_train_native_semantic_probe_balances_domains() -> None:
    cifake = _arrays([0, 0, 0, 1, 1, 1], offset=0.0)
    native = _arrays([0, 0, 1, 1], offset=0.25)

    model, summary = train_native_semantic_probe(cifake, native, seed=2026)

    assert summary == {
        "native_per_class": 2,
        "cifake_per_class": 2,
        "training_rows": 8,
    }
    assert model.n_features_in_ == 2
    assert model.predict_proba(cifake.semantic.astype(np.float32)).shape == (6, 2)


def test_train_native_semantic_probe_includes_aligned_augmented_views() -> None:
    cifake = _arrays([0, 0, 0, 1, 1, 1], offset=0.0)
    native = _arrays([0, 0, 1, 1], offset=0.25)
    cifake_augmented = FeatureArrays(
        **{
            **cifake.__dict__,
            "semantic": (cifake.semantic.astype(np.float32) + 0.1).astype(np.float16),
            "transform_keys": np.asarray(["jpeg_q70"] * len(cifake)),
        }
    )
    native_augmented = FeatureArrays(
        **{
            **native.__dict__,
            "semantic": (native.semantic.astype(np.float32) + 0.1).astype(np.float16),
            "transform_keys": np.asarray(["jpeg_q70"] * len(native)),
        }
    )

    _, summary = train_native_semantic_probe(
        cifake,
        native,
        seed=2026,
        cifake_augmented=cifake_augmented,
        native_augmented=native_augmented,
    )

    assert summary["training_rows"] == 16
    assert summary["views_per_image"] == 2


def test_train_native_semantic_probe_balances_wildfake_as_separate_domain() -> None:
    cifake = _arrays([0, 0, 0, 1, 1, 1], offset=0.0)
    sid = _arrays([0, 0, 1, 1], offset=0.25)
    wildfake = _arrays([0, 0, 0, 1, 1, 1], offset=0.5)
    augmented = []
    for arrays in (cifake, sid, wildfake):
        augmented.append(
            FeatureArrays(
                **{
                    **arrays.__dict__,
                    "semantic": (
                        arrays.semantic.astype(np.float32) + 0.1
                    ).astype(np.float16),
                    "transform_keys": np.asarray(["jpeg_q70"] * len(arrays)),
                }
            )
        )

    _, summary = train_native_semantic_probe(
        cifake,
        sid,
        seed=2026,
        cifake_augmented=augmented[0],
        native_augmented=augmented[1],
        wildfake=wildfake,
        wildfake_augmented=augmented[2],
    )

    assert summary["training_rows"] == 24
    assert summary["sid_per_class"] == 2
    assert summary["wildfake_per_class"] == 2
    assert summary["training_domains"] == ["cifake", "sid", "wildfake"]


def test_calibrated_threshold_and_gate_reject_constant_scores() -> None:
    arrays = _arrays([0, 0, 1, 1], offset=0.0)
    threshold = calibrate_threshold(
        arrays.labels, np.asarray([0.1, 0.2, 0.8, 0.9])
    )

    class ConstantModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            scores = np.full(len(features), 0.99)
            return np.column_stack((1.0 - scores, scores))

    result, failures = evaluate_gate(
        ConstantModel(), arrays, threshold=threshold, minimum_auc=0.7
    )

    assert result["passed"] is False
    assert any("score_std" in failure for failure in failures)
    assert any("predicted_fake_rate" in failure for failure in failures)


def test_validate_disjoint_rejects_cache_overlap() -> None:
    first = _arrays([0, 1], offset=0.0)
    second = _arrays([0, 1], offset=0.0)

    with np.testing.assert_raises_regex(ValueError, "overlap by 2 image IDs"):
        validate_disjoint(first, second, "train", "calibration")
