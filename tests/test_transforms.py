import numpy as np
import pytest
from PIL import Image

from src.detector.transforms import (
    EVAL_TRANSFORMS,
    apply_training_transform,
    apply_transform,
)


@pytest.fixture
def sample_image() -> Image.Image:
    values = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3) % 256
    return Image.fromarray(values.astype(np.uint8), mode="RGB")


@pytest.mark.parametrize("spec", EVAL_TRANSFORMS, ids=lambda spec: spec.key)
def test_transform_preserves_contract(sample_image: Image.Image, spec) -> None:
    transformed = apply_transform(sample_image, spec, seed=17)
    values = np.asarray(transformed)
    assert transformed.mode == "RGB"
    assert transformed.size == sample_image.size
    assert values.dtype == np.uint8
    assert values.min() >= 0
    assert values.max() <= 255


def test_seeded_transforms_are_deterministic(sample_image: Image.Image) -> None:
    for spec in EVAL_TRANSFORMS:
        first = np.asarray(apply_transform(sample_image, spec, seed=17))
        second = np.asarray(apply_transform(sample_image, spec, seed=17))
        np.testing.assert_array_equal(first, second)


def test_training_transform_is_deterministic(sample_image: Image.Image) -> None:
    first_image, first_spec = apply_training_transform(sample_image, 2026, "image-1")
    second_image, second_spec = apply_training_transform(sample_image, 2026, "image-1")
    assert first_spec == second_spec
    np.testing.assert_array_equal(np.asarray(first_image), np.asarray(second_image))
