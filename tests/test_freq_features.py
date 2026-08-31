import numpy as np
import pytest
from PIL import Image

from src.detector.freq_features import batch_radial_fft_features, radial_fft_features


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_radial_fft_features_are_finite(mode: str) -> None:
    shape = (32, 32) if mode == "L" else (32, 32, 3)
    image = Image.fromarray(np.zeros(shape, dtype=np.uint8), mode=mode)
    features = radial_fft_features(image)
    assert features.shape == (32,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_batch_radial_fft_features() -> None:
    images = [
        Image.new("RGB", (32, 32), color=(value, value, value)) for value in (0, 128)
    ]
    features = batch_radial_fft_features(images, bins=16, image_size=64)
    assert features.shape == (2, 16)
    assert np.isfinite(features).all()
