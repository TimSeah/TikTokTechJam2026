from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detector.features import (
    FFT_BINS,
    FFT_IMAGE_SIZE,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    FeatureArrays,
    load_feature_cache,
    read_cache_metadata,
)
from src.detector.model import (
    ARTIFACT_SCHEMA_VERSION,
    calculate_metrics,
    combine_features,
    predict_scores,
    save_artifact,
    validate_aligned,
)

FINAL_MODEL_NAME = "semantic_native_mixed"


def balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    random_generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"Requested {per_class} rows for label {label}, found {len(candidates)}"
            )
        selected.append(random_generator.choice(candidates, per_class, replace=False))
    return np.concatenate(selected)


def validate_disjoint(
    first: FeatureArrays,
    second: FeatureArrays,
    first_name: str,
    second_name: str,
) -> None:
    overlap = np.intersect1d(first.image_ids, second.image_ids)
    if len(overlap):
        examples = ", ".join(str(value) for value in overlap[:3])
        raise ValueError(
            f"{first_name} and {second_name} overlap by {len(overlap)} image IDs: "
            f"{examples}"
        )


def train_native_semantic_probe(
    cifake: FeatureArrays,
    native: FeatureArrays,
    seed: int,
    cifake_augmented: FeatureArrays | None = None,
    native_augmented: FeatureArrays | None = None,
    wildfake: FeatureArrays | None = None,
    wildfake_augmented: FeatureArrays | None = None,
) -> tuple[Pipeline, dict]:
    if (cifake_augmented is None) != (native_augmented is None):
        raise ValueError("Provide both augmented caches or neither")
    if (wildfake is None) != (wildfake_augmented is None):
        raise ValueError("Provide both WildFake caches or neither")
    if cifake_augmented is not None and native_augmented is not None:
        validate_aligned(cifake, cifake_augmented)
        validate_aligned(native, native_augmented)
    if wildfake is not None and wildfake_augmented is not None:
        validate_aligned(wildfake, wildfake_augmented)
    domains = {"cifake": cifake, "sid": native}
    augmented_domains = {
        "cifake": cifake_augmented,
        "sid": native_augmented,
    }
    if wildfake is not None:
        domains["wildfake"] = wildfake
        augmented_domains["wildfake"] = wildfake_augmented
    per_class = min(
        min(
            int(np.count_nonzero(arrays.labels == 0)),
            int(np.count_nonzero(arrays.labels == 1)),
        )
        for arrays in domains.values()
    )
    if per_class < 1:
        raise ValueError("Every training domain must contain both REAL and FAKE rows")
    indices = {
        name: balanced_indices(arrays.labels, per_class, seed)
        for name, arrays in domains.items()
    }
    feature_groups = [
        arrays.semantic[indices[name]].astype(np.float32, copy=False)
        for name, arrays in domains.items()
    ]
    label_groups = [
        arrays.labels[indices[name]] for name, arrays in domains.items()
    ]
    if cifake_augmented is not None and native_augmented is not None:
        feature_groups.extend(
            arrays.semantic[indices[name]].astype(np.float32, copy=False)
            for name, arrays in augmented_domains.items()
            if arrays is not None
        )
        label_groups.extend(
            arrays.labels[indices[name]]
            for name, arrays in augmented_domains.items()
            if arrays is not None
        )
    features = np.concatenate(feature_groups)
    labels = np.concatenate(label_groups)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(features, labels)
    summary = {
        "native_per_class": per_class,
        "cifake_per_class": per_class,
        "training_rows": len(labels),
    }
    if wildfake is not None:
        summary["sid_per_class"] = per_class
        summary["wildfake_per_class"] = per_class
        summary["training_domains"] = list(domains)
    if cifake_augmented is not None:
        summary["views_per_image"] = 2
        summary["augmentation"] = "clean_plus_deterministic_augmented"
    return pipeline, summary


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold calibration requires both classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    valid = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if not np.any(valid):
        raise ValueError("No finite probability threshold is available")
    valid_thresholds = thresholds[valid]
    objective = (true_positive_rate - false_positive_rate)[valid]
    best_objective = np.max(objective)
    best_indices = np.flatnonzero(np.isclose(objective, best_objective))
    best_index = min(
        best_indices, key=lambda index: abs(float(valid_thresholds[index]) - 0.5)
    )
    return float(valid_thresholds[best_index])


def evaluate_gate(
    model: Pipeline,
    arrays: FeatureArrays,
    threshold: float,
    minimum_auc: float,
    minimum_score_std: float = 0.01,
    minimum_fake_rate: float = 0.05,
    maximum_fake_rate: float = 0.95,
) -> tuple[dict, list[str]]:
    scores = predict_scores(model, arrays, semantic_only=True)
    metrics = calculate_metrics(arrays.labels, scores, threshold)
    predictions = scores >= threshold
    real_scores = scores[arrays.labels == 0]
    fake_scores = scores[arrays.labels == 1]
    if len(real_scores) == 0 or len(fake_scores) == 0:
        raise ValueError("Evaluation gate requires both classes")
    score_std = float(np.std(scores))
    predicted_fake_rate = float(np.mean(predictions))
    real_median = float(np.median(real_scores))
    fake_median = float(np.median(fake_scores))
    failures: list[str] = []
    if metrics.auc < minimum_auc:
        failures.append(f"auc {metrics.auc:.6f} < {minimum_auc:.6f}")
    if score_std < minimum_score_std:
        failures.append(f"score_std {score_std:.6f} < {minimum_score_std:.6f}")
    if not minimum_fake_rate <= predicted_fake_rate <= maximum_fake_rate:
        failures.append(
            f"predicted_fake_rate {predicted_fake_rate:.6f} outside "
            f"[{minimum_fake_rate:.6f}, {maximum_fake_rate:.6f}]"
        )
    if fake_median <= real_median:
        failures.append(
            f"fake_score_median {fake_median:.6f} <= real_score_median {real_median:.6f}"
        )
    return {
        "auc": metrics.auc,
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "score_std": score_std,
        "predicted_fake_rate": predicted_fake_rate,
        "real_score_median": real_median,
        "fake_score_median": fake_median,
        "minimum_auc": minimum_auc,
        "passed": not failures,
        "failures": failures,
    }, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a native-resolution semantic probe with balanced CIFAKE mixing."
    )
    parser.add_argument(
        "--cifake-train-cache", type=Path, default=Path("data/features/train-clean")
    )
    parser.add_argument(
        "--cifake-augmented-train-cache",
        type=Path,
        default=Path("data/features/train-augmented"),
    )
    parser.add_argument("--native-train-cache", type=Path, required=True)
    parser.add_argument("--native-augmented-train-cache", type=Path, required=True)
    parser.add_argument("--wildfake-train-cache", type=Path, required=True)
    parser.add_argument("--wildfake-augmented-train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument(
        "--cifake-eval-cache", type=Path, default=Path("data/features/test-clean")
    )
    parser.add_argument("--sid-eval-cache", type=Path, required=True)
    parser.add_argument("--wildfake-eval-cache", type=Path, required=True)
    parser.add_argument("--wildfake-matched-eval-cache", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/model-native.joblib")
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=Path("outputs/native_metrics.json")
    )
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--min-cifake-auc", type=float, default=0.90)
    parser.add_argument("--min-calibration-auc", type=float, default=0.90)
    parser.add_argument("--min-sid-auc", type=float, default=0.90)
    parser.add_argument("--min-wildfake-auc", type=float, default=0.70)
    parser.add_argument("--min-wildfake-matched-auc", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cifake_train = load_feature_cache(args.cifake_train_cache)
    cifake_augmented_train = load_feature_cache(args.cifake_augmented_train_cache)
    native_train = load_feature_cache(args.native_train_cache)
    native_augmented_train = load_feature_cache(args.native_augmented_train_cache)
    wildfake_train = load_feature_cache(args.wildfake_train_cache)
    wildfake_augmented_train = load_feature_cache(
        args.wildfake_augmented_train_cache
    )
    calibration = load_feature_cache(args.calibration_cache)
    cifake_eval = load_feature_cache(args.cifake_eval_cache)
    sid_eval = load_feature_cache(args.sid_eval_cache)
    wildfake_eval = load_feature_cache(args.wildfake_eval_cache)
    wildfake_matched_eval = load_feature_cache(args.wildfake_matched_eval_cache)
    validate_disjoint(native_train, calibration, "native train", "calibration")
    model, training_summary = train_native_semantic_probe(
        cifake_train,
        native_train,
        args.seed,
        cifake_augmented_train,
        native_augmented_train,
        wildfake_train,
        wildfake_augmented_train,
    )
    calibration_scores = predict_scores(model, calibration, semantic_only=True)
    threshold = calibrate_threshold(calibration.labels, calibration_scores)
    evaluation_specs = {
        "sid_calibration": (calibration, args.min_calibration_auc),
        "cifake_clean": (cifake_eval, args.min_cifake_auc),
        "sid_validation": (sid_eval, args.min_sid_auc),
        "wildfake_default": (wildfake_eval, args.min_wildfake_auc),
        "wildfake_laion_matched": (
            wildfake_matched_eval,
            args.min_wildfake_matched_auc,
        ),
    }
    evaluations: dict[str, dict] = {}
    gate_failures: dict[str, list[str]] = {}
    for name, (arrays, minimum_auc) in evaluation_specs.items():
        result, failures = evaluate_gate(model, arrays, threshold, minimum_auc)
        evaluations[name] = result
        if failures:
            gate_failures[name] = failures
    gates_passed = not gate_failures
    cifake_metadata = read_cache_metadata(args.cifake_train_cache)
    cifake_augmented_metadata = read_cache_metadata(args.cifake_augmented_train_cache)
    native_metadata = read_cache_metadata(args.native_train_cache)
    native_augmented_metadata = read_cache_metadata(args.native_augmented_train_cache)
    wildfake_metadata = read_cache_metadata(args.wildfake_train_cache)
    wildfake_augmented_metadata = read_cache_metadata(
        args.wildfake_augmented_train_cache
    )
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "final_model": FINAL_MODEL_NAME,
        "models": {FINAL_MODEL_NAME: model},
        "config": {
            "class_mapping": {"REAL": 0, "FAKE": 1},
            "score_class": "FAKE",
            "model_name": MODEL_NAME,
            "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
            "preprocessing": "OpenCLIP official evaluation preprocessing",
            "semantic_dimension": int(cifake_train.semantic.shape[1]),
            "frequency_dimension": 0,
            "fft_bins": FFT_BINS,
            "fft_image_size": FFT_IMAGE_SIZE,
            "final_feature_mode": "semantic",
            "threshold": threshold,
            "seed": args.seed,
            "training_sources": {
                "cifake_manifest_sha256": cifake_metadata["manifest_sha256"],
                "cifake_augmented_manifest_sha256": cifake_augmented_metadata[
                    "manifest_sha256"
                ],
                "native_manifest_sha256": native_metadata["manifest_sha256"],
                "native_augmented_manifest_sha256": native_augmented_metadata[
                    "manifest_sha256"
                ],
                "native_dataset": native_metadata.get("dataset"),
                "native_dataset_revision": native_metadata.get("dataset_revision"),
                "native_split": native_metadata.get("source_split"),
                "wildfake_manifest_sha256": wildfake_metadata["manifest_sha256"],
                "wildfake_augmented_manifest_sha256": wildfake_augmented_metadata[
                    "manifest_sha256"
                ],
                "wildfake_dataset": wildfake_metadata.get("dataset"),
                "wildfake_dataset_revision": wildfake_metadata.get(
                    "dataset_revision"
                ),
                "wildfake_split": wildfake_metadata.get("source_split"),
            },
            "promotion_gates_passed": gates_passed,
            **training_summary,
        },
    }
    save_artifact(artifact, args.output)
    metrics = {
        "model": FINAL_MODEL_NAME,
        "training": training_summary,
        "threshold": threshold,
        "gates_passed": gates_passed,
        "gate_failures": gate_failures,
        "evaluations": evaluations,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.metrics_out.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.metrics_out)
    if args.promote_to is not None:
        if not gates_passed:
            raise RuntimeError(
                f"Candidate failed promotion gates; existing model was not changed: "
                f"{gate_failures}"
            )
        save_artifact(artifact, args.promote_to)
    print(
        f"model={FINAL_MODEL_NAME} rows={training_summary['training_rows']} "
        f"threshold={threshold:.6f} gates_passed={gates_passed} output={args.output}"
    )


if __name__ == "__main__":
    main()
