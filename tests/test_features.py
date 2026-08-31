from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.detector.data import ImageRecord
from src.detector.features import (
    FeatureArrays,
    FeatureDataset,
    load_feature_cache,
    save_feature_shard,
)


def _preprocess(image: Image.Image) -> torch.Tensor:
    values = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1)


def test_feature_dataset_clean_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "train" / "REAL" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(image_path)
    record = ImageRecord("id", "train/REAL/sample.jpg", "train", "train", 0, "REAL")
    dataset = FeatureDataset([record], tmp_path, _preprocess, "clean", 2026)

    semantic_input, frequency, label, image_id, relative_path, transform_key = dataset[
        0
    ]
    assert semantic_input.shape == (3, 8, 8)
    assert frequency.shape == (32,)
    assert label == 0
    assert image_id == "id"
    assert relative_path == "train/REAL/sample.jpg"
    assert transform_key == "clean"


def test_feature_cache_round_trip(tmp_path: Path) -> None:
    arrays = FeatureArrays(
        semantic=np.ones((2, 512), dtype=np.float16),
        frequency=np.ones((2, 32), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int8),
        image_ids=np.asarray(["a", "b"]),
        relative_paths=np.asarray(["a.jpg", "b.jpg"]),
        transform_keys=np.asarray(["clean", "clean"]),
    )
    save_feature_shard(arrays, tmp_path / "part-00000.npz")
    loaded = load_feature_cache(tmp_path)
    np.testing.assert_array_equal(loaded.semantic, arrays.semantic)
    np.testing.assert_array_equal(loaded.frequency, arrays.frequency)
    np.testing.assert_array_equal(loaded.labels, arrays.labels)
    np.testing.assert_array_equal(loaded.image_ids, arrays.image_ids)
