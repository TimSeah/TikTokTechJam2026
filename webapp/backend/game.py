from __future__ import annotations

import io
import random
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from src.detector.transforms import (
    REPRESENTATIVE_TRANSFORMS,
    TransformSpec,
    apply_transform,
)

from .config import Settings
from .detector import DetectorService, Prediction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GAME_AUGMENTATIONS = REPRESENTATIVE_TRANSFORMS
AUGMENTATION_LABELS = {
    "jpeg_q50": "JPEG compression (quality 50)",
    "blur_sigma1.0": "Gaussian blur (sigma 1.0)",
    "resize_scale0.5": "Downscaled to 50%",
    "noise_sigma0.05": "Gaussian noise (5%)",
    "jitter_amount0.20": "Color jitter (20%)",
    "crop_ratio0.80": "Center crop (80%)",
}


@dataclass(frozen=True)
class ChallengeImage:
    path: Path
    label: str


@dataclass(frozen=True)
class ChallengeRound:
    image: ChallengeImage
    prediction: Prediction
    augmentation: TransformSpec
    augmentation_seed: int

    @property
    def augmentation_key(self) -> str:
        return self.augmentation.key

    @property
    def augmentation_label(self) -> str:
        return AUGMENTATION_LABELS[self.augmentation.key]


class ChallengeGame:
    def __init__(self, settings: Settings, detector: DetectorService) -> None:
        self.detector = detector
        self.cache_size = settings.round_cache_size
        self.images = tuple(
            image
            for root in settings.challenge_dataset_roots
            for image in self._discover_images(root)
        )
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

    @staticmethod
    def _augment_image(
        image_path: Path, augmentation: TransformSpec, seed: int
    ) -> Image.Image:
        with Image.open(image_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image).convert("RGB")
        return apply_transform(image, augmentation, seed)

    def create_round(self) -> tuple[str, ChallengeRound]:
        if not self.images:
            raise RuntimeError("No labeled REAL/FAKE challenge images were found")

        image = self._random.choice(self.images)
        augmentation = self._random.choice(GAME_AUGMENTATIONS)
        augmentation_seed = secrets.randbits(64)
        augmented_image = self._augment_image(
            image.path, augmentation, augmentation_seed
        )
        challenge_round = ChallengeRound(
            image=image,
            prediction=self.detector.predict_image(augmented_image),
            augmentation=augmentation,
            augmentation_seed=augmentation_seed,
        )
        round_id = secrets.token_urlsafe(18)
        with self._lock:
            self._rounds[round_id] = challenge_round
            while len(self._rounds) > self.cache_size:
                expired_round_id, _ = self._rounds.popitem(last=False)
                self._finished_rounds.discard(expired_round_id)
        return round_id, challenge_round

    def render_round_image(self, challenge_round: ChallengeRound) -> bytes:
        image = self._augment_image(
            challenge_round.image.path,
            challenge_round.augmentation,
            challenge_round.augmentation_seed,
        )
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()

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
