from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from src.detector.data import ImageRecord, write_manifest


def detector_records_from_provenance(manifest_path: Path) -> list[ImageRecord]:
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    records: list[ImageRecord] = []
    seen_ids: set[str] = set()
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Unsupported label {label} in {manifest_path}")
        class_name = "REAL" if label == 0 else "FAKE"
        if row["class_name"] != class_name:
            raise ValueError(
                f"Class/label mismatch for row {row.get('row_idx')} in {manifest_path}"
            )
        source_identifier = row.get("item_id") or row.get("img_id")
        if not source_identifier:
            raise ValueError(f"Missing source identifier in {manifest_path}")
        identity = "|".join(
            (
                row["dataset"],
                row.get("config", ""),
                row["split"],
                row["row_idx"],
                source_identifier,
            )
        )
        image_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        if image_id in seen_ids:
            raise ValueError(f"Duplicate source identity in {manifest_path}: {identity}")
        seen_ids.add(image_id)
        records.append(
            ImageRecord(
                image_id=image_id,
                relative_path=row["relative_path"],
                source_split=row["split"],
                role="evaluation",
                label=label,
                class_name=class_name,
            )
        )
    return sorted(records, key=lambda record: record.image_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive detector manifests from frozen native evaluation provenance."
    )
    parser.add_argument(
        "--blind-test-root", type=Path, default=Path("data/blind-test")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/manifests/native-eval")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "sid-validation": args.blind_test_root / "sid-validation" / "manifest.csv",
        "wildfake-default": args.blind_test_root
        / "wildfake-default"
        / "manifest.csv",
        "wildfake-laion-matched": args.blind_test_root
        / "wildfake-laion-matched"
        / "manifest.csv",
    }
    for name, source_path in sources.items():
        records = detector_records_from_provenance(source_path)
        output_path = args.output_dir / f"{name}.csv"
        write_manifest(records, output_path)
        real_count = sum(record.label == 0 for record in records)
        fake_count = sum(record.label == 1 for record in records)
        print(
            f"dataset={name} real={real_count} fake={fake_count} manifest={output_path}"
        )


if __name__ == "__main__":
    main()