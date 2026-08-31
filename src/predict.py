from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector.data import IMAGE_EXTENSIONS, ImageRecord
from src.detector.features import (
    FFT_BINS,
    FFT_IMAGE_SIZE,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    FeatureDataset,
    create_loader,
    encode_loader,
    load_backbone,
    resolve_device,
)
from src.detector.model import load_artifact, predict_scores


def discover_images(input_dir: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    skipped: list[str] = []
    paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in paths:
        relative_path = path.relative_to(input_dir).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            skipped.append(f"{relative_path}: {error}")
            continue
        records.append(
            ImageRecord(
                image_id=relative_path,
                relative_path=relative_path,
                source_split="inference",
                role="inference",
                label=-1,
                class_name="UNKNOWN",
            )
        )
    return records, skipped


def validate_artifact_config(config: dict) -> None:
    expected = {
        "model_name": MODEL_NAME,
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "fft_bins": FFT_BINS,
        "fft_image_size": FFT_IMAGE_SIZE,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Artifact feature configuration mismatch: {mismatches}")


def write_predictions(
    records: list[ImageRecord], scores, output_path: Path, threshold: float | None
) -> None:
    rows = []
    for record, score in zip(records, scores, strict=True):
        row = {"image_path": record.relative_path, "pred": float(score)}
        if threshold is not None:
            row["label"] = "FAKE" if score >= threshold else "REAL"
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict AI-generated image confidence."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include_label", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")
    artifact = load_artifact(args.model)
    config = artifact["config"]
    validate_artifact_config(config)
    records, skipped = discover_images(args.input_dir)
    for message in skipped:
        print(f"skip {message}", file=sys.stderr)
    if not records:
        write_predictions([], [], args.out, None)
        print("No valid supported images found", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    model, preprocess = load_backbone(device)
    dataset = FeatureDataset(
        records, args.input_dir, preprocess, "clean", config["seed"]
    )
    loader = create_loader(dataset, args.batch_size, args.workers, device)
    arrays = encode_loader(model, loader, device)
    expected_dimension = config["semantic_dimension"] + config["frequency_dimension"]
    actual_dimension = arrays.semantic.shape[1] + arrays.frequency.shape[1]
    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dimension}, got {actual_dimension}"
        )

    final_model = artifact["models"][artifact["final_model"]]
    scores = predict_scores(final_model, arrays, semantic_only=False)
    threshold = config["threshold"] if args.include_label else None
    write_predictions(records, scores, args.out, threshold)
    print(f"predicted={len(records)} skipped={len(skipped)} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
