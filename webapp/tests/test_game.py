from pathlib import Path

from PIL import Image

from webapp.backend.config import Settings
from webapp.backend.detector import Prediction
from webapp.backend.game import ChallengeGame


class StubDetector:
    def predict(self, image_path: Path) -> Prediction:
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


def test_unlabeled_images_are_excluded(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "unknown.png")
    assert ChallengeGame._discover_images(tmp_path) == ()
