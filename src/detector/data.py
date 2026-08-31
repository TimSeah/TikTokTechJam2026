from __future__ import annotations

import argparse
import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_LABELS = {"REAL": 0, "FAKE": 1}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    source_split: str
    role: str
    label: int
    class_name: str


def _image_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def discover_cifake(data_root: Path, split: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_name, label in CLASS_LABELS.items():
        class_dir = data_root / split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing CIFAKE class directory: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No supported images found in {class_dir}")
        for path in paths:
            relative_path = path.relative_to(data_root).as_posix()
            records.append(
                ImageRecord(
                    image_id=_image_id(relative_path),
                    relative_path=relative_path,
                    source_split=split,
                    role=split,
                    label=label,
                    class_name=class_name,
                )
            )
    return records


def balanced_sample(
    records: Iterable[ImageRecord], per_class: int, seed: int, role: str
) -> list[ImageRecord]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    random_generator = random.Random(seed)
    sampled: list[ImageRecord] = []
    records_list = list(records)
    for class_name, label in CLASS_LABELS.items():
        candidates = [record for record in records_list if record.label == label]
        if len(candidates) < per_class:
            raise ValueError(
                f"Requested {per_class} {class_name} images, found {len(candidates)}"
            )
        selected = random_generator.sample(candidates, per_class)
        sampled.extend(
            ImageRecord(
                image_id=record.image_id,
                relative_path=record.relative_path,
                source_split=record.source_split,
                role=role,
                label=record.label,
                class_name=record.class_name,
            )
            for record in selected
        )
    return sorted(sampled, key=lambda record: record.image_id)


def write_manifest(records: Iterable[ImageRecord], output_path: Path) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: Path) -> list[ImageRecord]:
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return [
        ImageRecord(
            image_id=row["image_id"],
            relative_path=row["relative_path"],
            source_split=row["source_split"],
            role=row["role"],
            label=int(row["label"]),
            class_name=row["class_name"],
        )
        for row in rows
    ]


def validate_images(
    records: Iterable[ImageRecord], data_root: Path, sample_size: int, seed: int
) -> None:
    records_list = list(records)
    random_generator = random.Random(seed)
    selected = random_generator.sample(
        records_list, min(sample_size, len(records_list))
    )
    for record in selected:
        image_path = data_root / record.relative_path
        with Image.open(image_path) as image:
            image.convert("RGB").load()


def build_manifests(
    data_root: Path,
    output_dir: Path,
    seed: int = 2026,
    dev_per_class: int = 100,
) -> dict[str, list[ImageRecord]]:
    train_records = discover_cifake(data_root, "train")
    test_records = discover_cifake(data_root, "test")
    dev_records = balanced_sample(
        train_records, per_class=dev_per_class, seed=seed, role="development"
    )

    manifests = {
        "train": train_records,
        "test": test_records,
        "development": dev_records,
    }
    for name, records in manifests.items():
        write_manifest(records, output_dir / f"{name}.csv")
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic CIFAKE manifests."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dev-per-class", type=int, default=100)
    parser.add_argument("--validate-sample", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = build_manifests(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        dev_per_class=args.dev_per_class,
    )
    for name, records in manifests.items():
        validate_images(records, args.data_root, args.validate_sample, args.seed)
        real_count = sum(record.label == 0 for record in records)
        fake_count = sum(record.label == 1 for record in records)
        print(f"{name}: total={len(records)} real={real_count} fake={fake_count}")


if __name__ == "__main__":
    main()
