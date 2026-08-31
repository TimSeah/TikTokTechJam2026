from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detector.features import FeatureArrays, load_feature_cache
from src.detector.freq_features import radial_fft_features
from src.detector.model import combine_features, load_artifact, predict_margins, predict_scores

HISTORICAL_REVISION = "ed8b1ef5eee8258c34e0589990c76765e470ea05"
HISTORICAL_ARTIFACT_PATH = "outputs/model.joblib"
HISTORICAL_ARTIFACT_SHA256 = (
    "e9bc59e42469c9f7001d7f23f6cbfbdac599e6968148329ef956faefc3427b5e"
)
BOOTSTRAP_REPLICATES = 5000
AUDIT_SEED = 2026

DATASETS = {
    "sid_validation": {
        "feature_cache": "data/features/sid-validation-clean",
        "manifest": "data/blind-test/sid-validation/manifest.csv",
    },
    "wildfake_coco_dalle": {
        "feature_cache": "data/features/wildfake-default-clean",
        "manifest": "data/blind-test/wildfake-default/manifest.csv",
    },
    "wildfake_laion_dalle": {
        "feature_cache": "data/features/wildfake-laion-matched-clean",
        "manifest": "data/blind-test/wildfake-laion-matched/manifest.csv",
    },
}

PROMOTED_GATES = {
    "sid_calibration": "data/features/sid-final-calibration",
    "sid_validation": "data/features/sid-validation-clean",
    "cifake_clean": "data/features/test-clean",
    "wildfake_coco_dalle": "data/features/wildfake-default-clean",
    "wildfake_laion_dalle": "data/features/wildfake-laion-matched-clean",
}


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def stratified_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = AUDIT_SEED,
) -> dict:
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("AUC bootstrap requires both classes")
    random_generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [
                random_generator.choice(indices, len(indices), replace=True)
                for indices in class_indices
            ]
        )
        estimates[index] = roc_auc_score(labels[sampled], scores[sampled])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "bootstrap_95_ci": percentile_interval(estimates),
        "bootstrap_replicates": replicates,
    }


def paired_bootstrap_auc_difference(
    labels: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = AUDIT_SEED,
) -> dict:
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Paired AUC bootstrap requires both classes")
    random_generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [
                random_generator.choice(indices, len(indices), replace=True)
                for indices in class_indices
            ]
        )
        differences[index] = roc_auc_score(
            labels[sampled], first_scores[sampled]
        ) - roc_auc_score(labels[sampled], second_scores[sampled])
    return {
        "difference": float(
            roc_auc_score(labels, first_scores)
            - roc_auc_score(labels, second_scores)
        ),
        "bootstrap_95_ci": percentile_interval(differences),
        "bootstrap_replicates": replicates,
    }


def reliability_diagnostics(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    expected_calibration_error = 0.0
    reliability = []
    for bin_index in range(bins):
        selected = assignments == bin_index
        count = int(np.count_nonzero(selected))
        if count:
            mean_score = float(np.mean(scores[selected]))
            positive_rate = float(np.mean(labels[selected]))
            expected_calibration_error += (
                count / len(labels) * abs(mean_score - positive_rate)
            )
        else:
            mean_score = None
            positive_rate = None
        reliability.append(
            {
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_score": mean_score,
                "positive_rate": positive_rate,
            }
        )
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "expected_calibration_error_10_bin": float(expected_calibration_error),
        "reliability_bins": reliability,
    }


def threshold_diagnostics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    predictions = scores >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "predicted_fake_rate": float(np.mean(predictions)),
        "confusion_matrix_tn_fp_fn_tp": [
            int(true_negative),
            int(false_positive),
            int(false_negative),
            int(true_positive),
        ],
    }


def quantile_summary(values: np.ndarray) -> dict:
    names = ("minimum", "p05", "median", "p95", "p99", "maximum")
    quantiles = np.percentile(values, [0, 5, 50, 95, 99, 100])
    return {name: float(value) for name, value in zip(names, quantiles, strict=True)}


def subsample_auc_sensitivity(
    labels: np.ndarray,
    scores: np.ndarray,
    per_class: int = 100,
    seeds: int = 20,
) -> dict:
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) < per_class for indices in class_indices):
        raise ValueError("Not enough examples for stratified subsampling")
    estimates = []
    for seed in range(AUDIT_SEED, AUDIT_SEED + seeds):
        random_generator = np.random.default_rng(seed)
        selected = np.concatenate(
            [
                random_generator.choice(indices, per_class, replace=False)
                for indices in class_indices
            ]
        )
        estimates.append(float(roc_auc_score(labels[selected], scores[selected])))
    return {
        "per_class": per_class,
        "seeds": list(range(AUDIT_SEED, AUDIT_SEED + seeds)),
        "aucs": estimates,
        "minimum": min(estimates),
        "median": float(np.median(estimates)),
        "maximum": max(estimates),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def metadata_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        width = float(row["actual_width"])
        height = float(row["actual_height"])
        features.append(
            [
                np.log1p(width),
                np.log1p(height),
                np.log1p(float(row["bytes"])),
                np.log(width / height),
            ]
        )
        labels.append(int(row["label"]))
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def metadata_baseline(rows: list[dict[str, str]], seeds: int = 5) -> dict:
    features, labels = metadata_matrix(rows)
    estimates = []
    for seed in range(AUDIT_SEED, AUDIT_SEED + seeds):
        predictions = np.empty(len(labels), dtype=np.float64)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_indices, test_indices in folds.split(features, labels):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=2000, random_state=seed),
                    ),
                ]
            )
            model.fit(features[train_indices], labels[train_indices])
            predictions[test_indices] = model.predict_proba(features[test_indices])[:, 1]
        estimates.append(float(roc_auc_score(labels, predictions)))
    return {
        "features": ["log_width", "log_height", "log_bytes", "log_aspect_ratio"],
        "protocol": "five-fold stratified out-of-fold logistic regression",
        "seeds": list(range(AUDIT_SEED, AUDIT_SEED + seeds)),
        "aucs": estimates,
        "mean_auc": float(np.mean(estimates)),
        "minimum_auc": min(estimates),
        "maximum_auc": max(estimates),
    }


def source_score_summary(
    rows: list[dict[str, str]],
    arrays: FeatureArrays,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    source_by_path = {
        row["relative_path"]: row.get("source") or "unspecified" for row in rows
    }
    sources = np.asarray(
        [source_by_path[str(relative_path)] for relative_path in arrays.relative_paths]
    )
    result = {}
    for source in sorted(np.unique(sources)):
        selected = sources == source
        source_labels = arrays.labels[selected]
        source_scores = scores[selected]
        label_counts = {
            str(label): int(np.count_nonzero(source_labels == label))
            for label in (0, 1)
        }
        result[str(source)] = {
            "count": int(np.count_nonzero(selected)),
            "label_counts": label_counts,
            "score": quantile_summary(source_scores),
            "predicted_fake_rate": float(np.mean(source_scores >= threshold)),
            "auc": (
                float(roc_auc_score(source_labels, source_scores))
                if len(np.unique(source_labels)) == 2
                else None
            ),
            "auc_note": (
                None
                if len(np.unique(source_labels)) == 2
                else "undefined because this source contains one class"
            ),
        }
    return result


def paired_wildfake_holdout(root: Path) -> dict:
    clean = load_feature_cache(root / "data/features/wildfake-final-clean")
    augmented = load_feature_cache(root / "data/features/wildfake-final-augmented")
    rows = read_csv_rows(root / "data/native-train/wildfake-final/provenance.csv")
    group_by_path = {
        row["relative_path"]: row["group"] for row in rows if row["role"] == "train"
    }
    groups = np.asarray(
        [group_by_path[str(relative_path)] for relative_path in clean.relative_paths]
    )
    real_groups = sorted(np.unique(groups[clean.labels == 0]))
    fake_groups = sorted(np.unique(groups[clean.labels == 1]))
    results = []
    for real_index, real_group in enumerate(real_groups):
        for fake_index, fake_group in enumerate(fake_groups):
            held_out = (groups == real_group) | (groups == fake_group)
            fitting = ~held_out
            features = np.concatenate(
                (
                    clean.semantic[fitting].astype(np.float32, copy=False),
                    augmented.semantic[fitting].astype(np.float32, copy=False),
                )
            )
            labels = np.concatenate((clean.labels[fitting], augmented.labels[fitting]))
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=AUDIT_SEED,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            model.fit(features, labels)
            held_out_arrays = FeatureArrays(
                semantic=clean.semantic[held_out],
                frequency=clean.frequency[held_out],
                labels=clean.labels[held_out],
                image_ids=clean.image_ids[held_out],
                relative_paths=clean.relative_paths[held_out],
                transform_keys=clean.transform_keys[held_out],
            )
            margins = predict_margins(model, held_out_arrays, semantic_only=True)
            result = stratified_bootstrap_auc(
                held_out_arrays.labels,
                margins,
                replicates=1000,
                seed=AUDIT_SEED + real_index * len(fake_groups) + fake_index,
            )
            result.update(
                {
                    "held_out_real_source": str(real_group),
                    "held_out_fake_generator": str(fake_group),
                    "evaluation_real": int(
                        np.count_nonzero(held_out_arrays.labels == 0)
                    ),
                    "evaluation_fake": int(
                        np.count_nonzero(held_out_arrays.labels == 1)
                    ),
                    "fitting_rows_clean_plus_augmented": int(2 * np.count_nonzero(fitting)),
                }
            )
            results.append(result)
    aucs = np.asarray([result["auc"] for result in results])
    return {
        "protocol": (
            "For each of 18 pairs, exclude one real source and one fake generator, "
            "fit a semantic logistic probe on clean plus augmented views of all "
            "remaining WildFake groups, and evaluate clean held-out images."
        ),
        "scope": (
            "Within the WildFake-Sample ecosystem; this sensitivity analysis is not "
            "an independent post-promotion test of the deployed model."
        ),
        "pairs": results,
        "summary": {
            "pair_count": len(results),
            "minimum_auc": float(np.min(aucs)),
            "median_auc": float(np.median(aucs)),
            "maximum_auc": float(np.max(aucs)),
        },
    }


def load_historical_artifact(root: Path) -> tuple[dict, str]:
    blob = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{HISTORICAL_REVISION}:{HISTORICAL_ARTIFACT_PATH}",
        ]
    )
    digest = hashlib.sha256(blob).hexdigest()
    if digest != HISTORICAL_ARTIFACT_SHA256:
        raise ValueError(f"Historical artifact hash mismatch: {digest}")
    return joblib.load(io.BytesIO(blob)), digest


def with_frequency_features(arrays: FeatureArrays, image_root: Path) -> FeatureArrays:
    frequency = []
    for relative_path in arrays.relative_paths:
        with Image.open(image_root / str(relative_path)) as opened_image:
            frequency.append(radial_fft_features(opened_image.convert("RGB")))
    return FeatureArrays(
        semantic=arrays.semantic,
        frequency=np.stack(frequency),
        labels=arrays.labels,
        image_ids=arrays.image_ids,
        relative_paths=arrays.relative_paths,
        transform_keys=arrays.transform_keys,
    )


def branch_diagnostics(model: Pipeline, arrays: FeatureArrays) -> dict:
    features = combine_features(arrays, semantic_only=False)
    standardized = model.named_steps["scaler"].transform(features)
    coefficients = model.named_steps["classifier"].coef_[0]
    contributions = standardized * coefficients
    semantic = np.sum(contributions[:, : arrays.semantic.shape[1]], axis=1)
    frequency = np.sum(contributions[:, arrays.semantic.shape[1] :], axis=1)
    absolute_frequency = np.abs(standardized[:, arrays.semantic.shape[1] :])
    return {
        "semantic_logit_contribution": {
            "all": quantile_summary(semantic),
            "real": quantile_summary(semantic[arrays.labels == 0]),
            "fake": quantile_summary(semantic[arrays.labels == 1]),
        },
        "frequency_logit_contribution": {
            "all": quantile_summary(frequency),
            "real": quantile_summary(frequency[arrays.labels == 0]),
            "fake": quantile_summary(frequency[arrays.labels == 1]),
        },
        "absolute_standardized_frequency": quantile_summary(
            absolute_frequency.reshape(-1)
        ),
    }


def score_audit(
    model: Pipeline,
    arrays: FeatureArrays,
    semantic_only: bool,
    threshold: float,
    manifest_rows: list[dict[str, str]] | None = None,
) -> dict:
    probabilities = predict_scores(model, arrays, semantic_only)
    margins = predict_margins(model, arrays, semantic_only)
    result = {
        "probability_ranking": stratified_bootstrap_auc(arrays.labels, probabilities),
        "margin_ranking": stratified_bootstrap_auc(arrays.labels, margins),
        "margin_minus_probability_auc": paired_bootstrap_auc_difference(
            arrays.labels, margins, probabilities
        ),
        "probability_saturation": {
            "exact_zero": int(np.count_nonzero(probabilities == 0.0)),
            "exact_one": int(np.count_nonzero(probabilities == 1.0)),
            "unique_probabilities": int(len(np.unique(probabilities))),
            "unique_margins": int(len(np.unique(margins))),
        },
        "threshold_metrics": threshold_diagnostics(
            arrays.labels, probabilities, threshold
        ),
        "calibration": reliability_diagnostics(arrays.labels, probabilities),
        "subsample_sensitivity": subsample_auc_sensitivity(
            arrays.labels, margins
        ),
    }
    if not semantic_only:
        result["branches"] = branch_diagnostics(model, arrays)
    if manifest_rows is not None:
        result["sources"] = source_score_summary(
            manifest_rows, arrays, probabilities, threshold
        )
    return result


def duplicate_audit(root: Path) -> dict:
    training_images: list[tuple[Path, int, str, str | None]] = []
    for row in read_csv_rows(root / "data/manifests/train.csv"):
        training_images.append(
            (
                root / "data/downloads" / row["relative_path"],
                int(row["label"]),
                "cifake",
                None,
            )
        )
    for dataset, relative_root, provenance_path in (
        (
            "sid",
            "data/native-train/sid-final",
            "data/native-train/sid-final/provenance.csv",
        ),
        (
            "wildfake",
            "data/native-train/wildfake-final",
            "data/native-train/wildfake-final/provenance.csv",
        ),
    ):
        for row in read_csv_rows(root / provenance_path):
            if row["role"] == "train":
                training_images.append(
                    (
                        root / relative_root / row["relative_path"],
                        int(row["label"]),
                        dataset,
                        row["sha256"],
                    )
                )

    exact_hashes: set[str] = set()
    perceptual_labels: dict[int, set[int]] = {}
    perceptual_sources: dict[int, set[str]] = {}
    for path, label, dataset, known_sha256 in training_images:
        digest = known_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
        exact_hashes.add(digest)
        difference_hash = image_difference_hash(path)
        perceptual_labels.setdefault(difference_hash, set()).add(label)
        perceptual_sources.setdefault(difference_hash, set()).add(dataset)
    near_index = build_near_hash_index(perceptual_labels)

    result = {
        "method": (
            "Exact SHA-256 plus 64-bit horizontal difference hash. The five-band "
            "candidate index exhaustively finds every Hamming-distance <= 4 match; "
            "difference hashes are a screening tool, not semantic duplicate proof."
        ),
        "training_images": len(training_images),
        "training_unique_sha256": len(exact_hashes),
        "training_unique_difference_hashes": len(perceptual_labels),
        "evaluations": {},
    }
    for dataset_name, spec in DATASETS.items():
        rows = read_csv_rows(root / spec["manifest"])
        hashes = [row["sha256"] for row in rows]
        near_matches = 0
        opposite_label_matches = 0
        matched_training_sources: set[str] = set()
        for row in rows:
            image_hash = image_difference_hash(
                root / "data/blind-test" / row["relative_path"]
            )
            candidates = near_hash_candidates(near_index, image_hash)
            matches = [
                candidate
                for candidate in candidates
                if (candidate ^ image_hash).bit_count() <= 4
            ]
            if matches:
                near_matches += 1
                evaluation_label = int(row["label"])
                if any(
                    evaluation_label not in perceptual_labels[candidate]
                    for candidate in matches
                ):
                    opposite_label_matches += 1
                for candidate in matches:
                    matched_training_sources.update(perceptual_sources[candidate])
        result["evaluations"][dataset_name] = {
            "rows": len(rows),
            "unique_sha256": len(set(hashes)),
            "within_evaluation_duplicates": len(hashes) - len(set(hashes)),
            "training_sha256_overlaps": len(exact_hashes.intersection(hashes)),
            "near_difference_hash_matches_hamming_le_4": near_matches,
            "near_matches_with_opposite_training_label": opposite_label_matches,
            "matched_training_sources": sorted(matched_training_sources),
        }
    return result


def image_difference_hash(path: Path) -> int:
    with Image.open(path) as opened_image:
        grayscale = ImageOps.exif_transpose(opened_image).convert("L")
        resized = grayscale.resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.uint8)
    comparisons = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    result = 0
    for bit in comparisons:
        result = (result << 1) | int(bit)
    return result


def hash_bands(value: int) -> list[tuple[int, int]]:
    bands = []
    offset = 0
    for band_index, width in enumerate((13, 13, 13, 13, 12)):
        bands.append((band_index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return bands


def build_near_hash_index(values: dict[int, set[int]]) -> dict[tuple[int, int], set[int]]:
    index: dict[tuple[int, int], set[int]] = {}
    for value in values:
        for band in hash_bands(value):
            index.setdefault(band, set()).add(value)
    return index


def near_hash_candidates(
    index: dict[tuple[int, int], set[int]], value: int
) -> set[int]:
    candidates: set[int] = set()
    for band in hash_bands(value):
        candidates.update(index.get(band, set()))
    return candidates


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit blind-set ranking, calibration, metadata, and overlap"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/shift_audit.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    historical_artifact, historical_hash = load_historical_artifact(root)
    promoted_artifact = load_artifact(root / "outputs/model.joblib")
    historical_results = {}
    metadata_results = {}
    for dataset_index, (dataset_name, spec) in enumerate(DATASETS.items()):
        semantic_arrays = load_feature_cache(root / spec["feature_cache"])
        manifest_rows = read_csv_rows(root / spec["manifest"])
        arrays = with_frequency_features(
            semantic_arrays, root / "data/blind-test"
        )
        models = {}
        for model_index, (model_name, model) in enumerate(
            historical_artifact["models"].items()
        ):
            models[model_name] = score_audit(
                model,
                arrays,
                semantic_only=model_name == "semantic_clean",
                threshold=0.5,
                manifest_rows=manifest_rows,
            )
        historical_results[dataset_name] = models
        metadata_results[dataset_name] = metadata_baseline(
            manifest_rows
        )

    promoted_model = promoted_artifact["models"][promoted_artifact["final_model"]]
    promoted_threshold = float(promoted_artifact["config"]["threshold"])
    promoted_results = {}
    for gate_name, cache_path in PROMOTED_GATES.items():
        arrays = load_feature_cache(root / cache_path)
        promoted_results[gate_name] = score_audit(
            promoted_model, arrays, semantic_only=True, threshold=promoted_threshold
        )

    payload = {
        "schema_version": 1,
        "protocol": {
            "historical_artifact_git_revision": HISTORICAL_REVISION,
            "historical_artifact_path": HISTORICAL_ARTIFACT_PATH,
            "historical_artifact_sha256": historical_hash,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_method": "stratified percentile bootstrap",
            "audit_seed": AUDIT_SEED,
            "subsample_note": (
                "Twenty deterministic half-samples quantify sensitivity within each "
                "frozen manifest; they are not independent dataset draws."
            ),
        },
        "historical_blind_models": historical_results,
        "promoted_semantic": promoted_results,
        "metadata_only_baselines": metadata_results,
        "paired_wildfake_group_holdout": paired_wildfake_holdout(root),
        "exact_duplicate_audit": duplicate_audit(root),
        "independent_post_promotion_test": {
            "status": "not_available",
            "reason": (
                "All locally available native datasets influenced fitting, calibration, "
                "or promotion. Reusing them cannot create a post-promotion blind test."
            ),
            "preregistered_requirements": [
                "lock model artifact and threshold before sampling",
                "use generators and real-image sources absent from fitting and promotion",
                "match content, dimensions, format, and compression where feasible",
                "report raw-margin and probability AUC with 95% intervals",
                "report saturation, calibration, metadata baselines, and per-source results",
            ],
        },
    }
    output_path = args.output if args.output.is_absolute() else root / args.output
    write_json_atomic(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()