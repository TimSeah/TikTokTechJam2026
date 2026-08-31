from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from src.detector.transforms import TransformSpec
from webapp.backend import game as game_module
from webapp.backend.config import Settings
from webapp.backend.detector import Prediction
from webapp.backend.game import ChallengeGame
from webapp.backend.main import create_round, get_round_image


class StubDetector:
    def __init__(self) -> None:
        self.image: Image.Image | None = None

    def predict(self, image_path: Path) -> Prediction:
        return Prediction(fake_probability=0.9, label="FAKE", elapsed_ms=12.5)

    def predict_image(self, image: Image.Image) -> Prediction:
        self.image = image.copy()
        return Prediction(fake_probability=0.9, label="FAKE", elapsed_ms=12.5)


def test_round_hides_label_until_finished(tmp_path: Path) -> None:
    real_dir = tmp_path / "REAL"
    real_dir.mkdir()
    Image.new("RGB", (8, 8)).save(real_dir / "sample.png")
    settings = Settings(
        model_artifact_path=tmp_path / "model.joblib",
        model_device="cpu",
        challenge_dataset_path=tmp_path,
        allowed_origins=(),
        round_cache_size=10,
    )

    game = ChallengeGame(settings, StubDetector())  # type: ignore[arg-type]
    round_id, challenge_round = game.create_round()

    assert challenge_round.image.label == "REAL"
    assert game.get_round(round_id) == challenge_round
    assert game.finish_round(round_id) == challenge_round
    assert game.finish_round(round_id) is None
    assert game.get_round(round_id) == challenge_round


def test_round_serves_same_augmented_pixels_scored_by_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_dir = tmp_path / "REAL"
    real_dir.mkdir()
    source = Image.linear_gradient("L").convert("RGB").resize((32, 32))
    source.save(real_dir / "sample.png")
    monkeypatch.setattr(
        game_module,
        "GAME_AUGMENTATIONS",
        (TransformSpec("blur", "sigma1.0", 1.0),),
    )
    settings = Settings(
        model_artifact_path=tmp_path / "model.joblib",
        model_device="cpu",
        challenge_dataset_path=tmp_path,
        allowed_origins=(),
        round_cache_size=10,
    )
    detector = StubDetector()
    game = ChallengeGame(settings, detector)  # type: ignore[arg-type]

    round_payload = create_round(game)
    challenge_round = game.get_round(round_payload["round_id"])
    assert challenge_round is not None
    image_response = get_round_image(round_payload["round_id"], game)
    with Image.open(BytesIO(image_response.body)) as rendered:
        rendered_image = rendered.convert("RGB")

    assert round_payload["augmentation"] == {
        "key": "blur_sigma1.0",
        "label": "Gaussian blur (sigma 1.0)",
    }
    assert image_response.media_type == "image/png"
    assert image_response.headers["cache-control"] == "no-store"
    assert detector.image is not None
    assert ImageChops.difference(detector.image, rendered_image).getbbox() is None


def test_multiple_dataset_roots_are_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wildfake_root = tmp_path / "wildfake"
    sid_root = tmp_path / "sid"
    for root, label in ((wildfake_root, "REAL"), (sid_root, "FAKE")):
        label_dir = root / label
        label_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(label_dir / "sample.png")
    monkeypatch.setenv(
        "CHALLENGE_DATASET_PATHS", f"{wildfake_root},{sid_root}"
    )

    settings = Settings.from_env()
    game = ChallengeGame(settings, StubDetector())  # type: ignore[arg-type]

    assert settings.challenge_dataset_paths == (
        wildfake_root.resolve(),
        sid_root.resolve(),
    )
    assert {(image.path, image.label) for image in game.images} == {
        ((wildfake_root / "REAL" / "sample.png").resolve(), "REAL"),
        ((sid_root / "FAKE" / "sample.png").resolve(), "FAKE"),
    }


def test_unlabeled_images_are_excluded(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "unknown.png")
    assert ChallengeGame._discover_images(tmp_path) == ()
