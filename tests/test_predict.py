import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.predict import discover_images, write_predictions


def test_discovery_skips_corrupt_images_and_sorts_paths(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    (tmp_path / "broken.webp").write_text("not an image", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    records, skipped = discover_images(tmp_path)
    assert [record.relative_path for record in records] == ["a.jpg", "b.png"]
    assert len(skipped) == 1
    assert skipped[0].startswith("broken.webp:")


def test_default_prediction_schema(tmp_path: Path) -> None:
    records, _ = discover_images(tmp_path)
    assert records == []

    from src.detector.data import ImageRecord

    record = ImageRecord("a", "a.jpg", "inference", "inference", -1, "UNKNOWN")
    output_path = tmp_path / "nested" / "predictions.json"
    write_predictions([record], np.asarray([0.75]), output_path, None)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [{"image_path": "a.jpg", "pred": 0.75}]
