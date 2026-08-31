from __future__ import annotations

import random
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .detector import DetectorService, Prediction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ChallengeImage:
    path: Path
    label: str


@dataclass(frozen=True)
class ChallengeRound:
    image: ChallengeImage
    prediction: Prediction


class ChallengeGame:
    def __init__(self, settings: Settings, detector: DetectorService) -> None:
        self.detector = detector
        self.cache_size = settings.round_cache_size
        self.images = self._discover_images(settings.challenge_dataset_path)
        self._rounds: OrderedDict[str, ChallengeRound] = OrderedDict()
        self._finished_rounds: set[str] = set()
        self._lock = threading.Lock()
        self._random = random.SystemRandom()

    @staticmethod
    def _discover_images(dataset_path: Path) -> tuple[ChallengeImage, ...]:
        if not dataset_path.is_dir():
            return ()

        images: list[ChallengeImage] = []
        for path in dataset_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            labels = {part.upper() for part in path.relative_to(dataset_path).parts}
            label = "FAKE" if "FAKE" in labels else "REAL" if "REAL" in labels else None
            if label is not None:
                images.append(ChallengeImage(path=path.resolve(), label=label))
        return tuple(images)

    def create_round(self) -> tuple[str, ChallengeRound]:
        if not self.images:
            raise RuntimeError("No labeled REAL/FAKE challenge images were found")

        image = self._random.choice(self.images)
        challenge_round = ChallengeRound(
            image=image,
            prediction=self.detector.predict(image.path),
        )
        round_id = secrets.token_urlsafe(18)
        with self._lock:
            self._rounds[round_id] = challenge_round
            while len(self._rounds) > self.cache_size:
                expired_round_id, _ = self._rounds.popitem(last=False)
                self._finished_rounds.discard(expired_round_id)
        return round_id, challenge_round

    def get_round(self, round_id: str) -> ChallengeRound | None:
        with self._lock:
            return self._rounds.get(round_id)

    def finish_round(self, round_id: str) -> ChallengeRound | None:
        with self._lock:
            if round_id in self._finished_rounds:
                return None
            challenge_round = self._rounds.get(round_id)
            if challenge_round is not None:
                self._finished_rounds.add(round_id)
            return challenge_round
