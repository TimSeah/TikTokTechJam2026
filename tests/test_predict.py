import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.detector.features import (
    FFT_BINS,
    FFT_IMAGE_SIZE,
    MODEL_NAME,
    PRETRAINED_CHECKPOINT,
)
from src.predict import discover_images, validate_artifact_config, write_predictions


def _artifact_config() -> dict:
    return {
        "model_name": MODEL_NAME,
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "fft_bins": FFT_BINS,
        "fft_image_size": FFT_IMAGE_SIZE,
    }


def test_artifact_config_defaults_to_legacy_hybrid_mode() -> None:
    validate_artifact_config(_artifact_config())


def test_semantic_artifact_does_not_require_fft_config() -> None:
    config = _artifact_config()
    config["final_feature_mode"] = "semantic"
    del config["fft_bins"]
    del config["fft_image_size"]

    validate_artifact_config(config)


def test_artifact_config_rejects_unknown_feature_mode() -> None:
    config = _artifact_config()
    config["final_feature_mode"] = "pixels"

    with pytest.raises(ValueError, match="Unsupported final feature mode"):
        validate_artifact_config(config)


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
