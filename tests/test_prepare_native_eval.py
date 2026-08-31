import csv
from pathlib import Path

import pytest

from src.detector.prepare_native_eval import detector_records_from_provenance


def _write_provenance(path: Path, class_name: str = "REAL") -> None:
    row = {
        "dataset": "example/dataset",
        "config": "default",
        "split": "validation",
        "row_idx": "17",
        "item_id": "source-17",
        "label": "0",
        "class_name": class_name,
        "relative_path": "dataset/REAL/image.jpg",
    }
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_detector_records_from_provenance_assigns_evaluation_role(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_provenance(manifest_path)

    records = detector_records_from_provenance(manifest_path)

    assert len(records) == 1
    assert records[0].role == "evaluation"
    assert records[0].source_split == "validation"
    assert records[0].relative_path == "dataset/REAL/image.jpg"


def test_detector_records_from_provenance_rejects_label_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_provenance(manifest_path, class_name="FAKE")

    with pytest.raises(ValueError, match="Class/label mismatch"):
        detector_records_from_provenance(manifest_path)