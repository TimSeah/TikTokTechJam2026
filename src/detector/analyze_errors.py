from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.detector.features import load_feature_cache, parse_condition
from src.detector.model import load_artifact, predict_scores, resolve_feature_mode
from src.detector.transforms import TransformSpec, apply_transform, stable_seed


@dataclass(frozen=True)
class ErrorExample:
    index: int
    error_type: str
    score: float


def select_errors(
    labels: np.ndarray, scores: np.ndarray, per_type: int, threshold: float
) -> list[ErrorExample]:
    false_positive_indices = np.flatnonzero((labels == 0) & (scores >= threshold))
    false_negative_indices = np.flatnonzero((labels == 1) & (scores < threshold))
    false_positive_indices = false_positive_indices[
        np.argsort(scores[false_positive_indices])[::-1]
    ][:per_type]
    false_negative_indices = false_negative_indices[
        np.argsort(scores[false_negative_indices])
    ][:per_type]
    examples = [
        ErrorExample(int(index), "false_positive", float(scores[index]))
        for index in false_positive_indices
    ]
    examples.extend(
        ErrorExample(int(index), "false_negative", float(scores[index]))
        for index in false_negative_indices
    )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export representative detector errors."
    )
    parser.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/downloads"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/error_examples")
    )
    parser.add_argument("--per-type", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_type < 1:
        raise ValueError("--per-type must be positive")
    artifact = load_artifact(args.model)
    arrays = load_feature_cache(args.cache)
    final_model = artifact["models"][artifact["final_model"]]
    semantic_only = resolve_feature_mode(artifact["config"]) == "semantic"
    scores = predict_scores(final_model, arrays, semantic_only=semantic_only)
    threshold = artifact["config"]["threshold"]
    examples = select_errors(arrays.labels, scores, args.per_type, threshold)
    if len(examples) != 2 * args.per_type:
        raise RuntimeError("Not enough false positives and false negatives to export")

    condition = parse_condition(args.condition)
    if not isinstance(condition, TransformSpec):
        raise ValueError("Error export requires a transformed evaluation condition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counters = {"false_positive": 0, "false_negative": 0}
    for example in examples:
        counters[example.error_type] += 1
        number = counters[example.error_type]
        source_path = args.data_root / arrays.relative_paths[example.index]
        image_id = str(arrays.image_ids[example.index])
        transform_seed = stable_seed(
            artifact["config"]["seed"], image_id, condition.key
        )
        with Image.open(source_path) as opened_image:
            transformed = apply_transform(opened_image, condition, transform_seed)
        output_name = f"{example.error_type}_{number}.jpg"
        transformed.save(args.output_dir / output_name, quality=95)
        rows.append(
            {
                "error_type": example.error_type,
                "rank": number,
                "image": output_name,
                "source_path": arrays.relative_paths[example.index],
                "image_id": image_id,
                "true_label": int(arrays.labels[example.index]),
                "fake_score": example.score,
                "condition": condition.key,
                "threshold": threshold,
            }
        )

    metadata_path = args.output_dir / "examples.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['error_type']} rank={row['rank']} score={row['fake_score']:.6f} "
            f"source={row['source_path']}"
        )


if __name__ == "__main__":
    main()
