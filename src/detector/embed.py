from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.detector.data import ImageRecord, read_manifest
from src.detector.features import (
    CACHE_SCHEMA_VERSION,
    FFT_BINS,
    FFT_IMAGE_SIZE,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
    FeatureDataset,
    create_loader,
    encode_loader,
    load_backbone,
    manifest_sha256,
    parse_condition,
    read_cache_metadata,
    resolve_device,
    save_feature_shard,
    write_cache_metadata,
)


def balanced_limit(records: list[ImageRecord], limit: int | None) -> list[ImageRecord]:
    if limit is None or limit >= len(records):
        return records
    if limit < 2:
        raise ValueError("--limit must be at least 2")
    per_class = limit // 2
    selected = [record for record in records if record.label == 0][:per_class]
    selected += [record for record in records if record.label == 1][:per_class]
    return sorted(selected, key=lambda record: record.image_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP and FFT feature shards.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--semantic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.shard_size < 1:
        raise ValueError(
            "Batch size and shard size must be positive; workers cannot be negative"
        )

    records = balanced_limit(read_manifest(args.manifest), args.limit)
    condition = parse_condition(args.condition)
    condition_key = condition if isinstance(condition, str) else condition.key
    device = resolve_device(args.device)
    manifest_hash = manifest_sha256(args.manifest)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": False,
        "condition": condition_key,
        "count": len(records),
        "manifest_sha256": manifest_hash,
        "model_name": MODEL_NAME,
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "fft_bins": FFT_BINS,
        "fft_image_size": FFT_IMAGE_SIZE,
        "seed": args.seed,
        "shard_size": args.shard_size,
    }
    if args.semantic_only:
        metadata["feature_mode"] = "semantic"
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists():
        existing = read_cache_metadata(args.output_dir)
        comparable = {
            key: value for key, value in existing.items() if key != "complete"
        }
        expected = {key: value for key, value in metadata.items() if key != "complete"}
        if comparable != expected:
            raise RuntimeError(f"Cache metadata mismatch in {args.output_dir}")
    else:
        write_cache_metadata(args.output_dir, metadata)

    print(f"device={device} condition={condition_key} images={len(records)}")
    model, preprocess = load_backbone(device)
    shard_count = math.ceil(len(records) / args.shard_size)
    for shard_index in range(shard_count):
        output_path = args.output_dir / f"part-{shard_index:05d}.npz"
        if output_path.exists():
            print(f"skip existing {output_path}")
            continue
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(records))
        dataset = FeatureDataset(
            records[start:end],
            args.data_root,
            preprocess,
            condition,
            args.seed,
            semantic_only=args.semantic_only,
        )
        loader = create_loader(dataset, args.batch_size, args.workers, device)
        arrays = encode_loader(model, loader, device)
        save_feature_shard(arrays, output_path)
        print(f"wrote {output_path} rows={len(arrays)}")

    metadata["complete"] = True
    write_cache_metadata(args.output_dir, metadata)
    print(f"complete rows={len(records)} shards={shard_count}")


if __name__ == "__main__":
    main()
