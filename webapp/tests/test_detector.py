import numpy as np
from PIL import Image

from webapp.backend import detector


def test_semantic_features_bypass_fft(monkeypatch) -> None:
    semantic = np.ones((1, 512), dtype=np.float32)
    config = {"final_feature_mode": "semantic"}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("FFT extraction should be bypassed")

    monkeypatch.setattr(detector, "_radial_fft_features", fail_if_called)

    actual = detector._classifier_features(semantic, Image.new("RGB", (8, 8)), config)

    assert actual is semantic


def test_missing_feature_mode_preserves_legacy_hybrid_behavior() -> None:
    semantic = np.ones((1, 2), dtype=np.float32)
    config = {"fft_bins": 3, "fft_image_size": 8}

    actual = detector._classifier_features(semantic, Image.new("RGB", (8, 8)), config)

    assert actual.shape == (1, 5)
