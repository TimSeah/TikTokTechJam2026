from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.detector.features import (
    FFT_BINS,
    FFT_IMAGE_SIZE,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    load_feature_cache,
    read_cache_metadata,
)
from src.detector.model import (
    ARTIFACT_SCHEMA_VERSION,
    FINAL_MODEL_NAME,
    MODEL_NAMES,
    save_artifact,
    score_models,
    train_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detector probe ablations.")
    parser.add_argument("--clean-train-cache", type=Path, required=True)
    parser.add_argument("--augmented-train-cache", type=Path, required=True)
    parser.add_argument("--clean-eval-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_train = load_feature_cache(args.clean_train_cache)
    augmented_train = load_feature_cache(args.augmented_train_cache)
    clean_eval = load_feature_cache(args.clean_eval_cache)
    models = train_models(clean_train, augmented_train, args.seed)
    metrics = score_models(models, clean_eval, args.threshold)
    clean_metadata = read_cache_metadata(args.clean_train_cache)
    augmented_metadata = read_cache_metadata(args.augmented_train_cache)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "final_model": FINAL_MODEL_NAME,
        "models": models,
        "config": {
            "class_mapping": {"REAL": 0, "FAKE": 1},
            "score_class": "FAKE",
            "model_name": MODEL_NAME,
            "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
            "preprocessing": "open_clip model defaults; RGB conversion; normalized CLIP embedding",
            "semantic_dimension": int(clean_train.semantic.shape[1]),
            "frequency_dimension": int(clean_train.frequency.shape[1]),
            "fft_bins": FFT_BINS,
            "fft_image_size": FFT_IMAGE_SIZE,
            "fft_recipe": "grayscale; bicubic 256x256; mean-center; Hann window; log1p magnitude; 32 radial bins",
            "threshold": args.threshold,
            "seed": args.seed,
            "train_manifest_sha256": clean_metadata["manifest_sha256"],
            "augmentation_condition": augmented_metadata["condition"],
            "model_names": list(MODEL_NAMES),
        },
    }
    save_artifact(artifact, args.output)

    metrics_payload = {
        "evaluation_condition": "clean",
        "threshold": args.threshold,
        "models": {name: asdict(result) for name, result in metrics.items()},
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    for name, result in metrics.items():
        print(
            f"{name}: auc={result.auc:.6f} accuracy={result.accuracy:.6f} "
            f"f1={result.f1:.6f}"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
