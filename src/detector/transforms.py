from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class TransformSpec:
    family: str
    severity: str
    value: float

    @property
    def key(self) -> str:
        return f"{self.family}_{self.severity}"


EVAL_TRANSFORMS = (
    TransformSpec("jpeg", "q90", 90),
    TransformSpec("jpeg", "q70", 70),
    TransformSpec("jpeg", "q50", 50),
    TransformSpec("jpeg", "q30", 30),
    TransformSpec("blur", "sigma0.5", 0.5),
    TransformSpec("blur", "sigma1.0", 1.0),
    TransformSpec("blur", "sigma2.0", 2.0),
    TransformSpec("resize", "scale0.5", 0.5),
    TransformSpec("resize", "scale0.25", 0.25),
    TransformSpec("noise", "sigma0.02", 0.02),
    TransformSpec("noise", "sigma0.05", 0.05),
    TransformSpec("noise", "sigma0.10", 0.10),
    TransformSpec("jitter", "amount0.20", 0.20),
    TransformSpec("crop", "ratio0.80", 0.80),
)

REPRESENTATIVE_TRANSFORMS = (
    EVAL_TRANSFORMS[2],
    EVAL_TRANSFORMS[5],
    EVAL_TRANSFORMS[7],
    EVAL_TRANSFORMS[10],
    EVAL_TRANSFORMS[12],
    EVAL_TRANSFORMS[13],
)


def stable_seed(global_seed: int, image_id: str, transform_key: str) -> int:
    payload = f"{global_seed}:{image_id}:{transform_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def _resize(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    downsampled_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    downsampled = image.resize(downsampled_size, Image.Resampling.BICUBIC)
    return downsampled.resize((width, height), Image.Resampling.BICUBIC)


def _noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    values = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, values.shape)
    transformed = np.clip(values + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(transformed * 255.0).astype(np.uint8), mode="RGB")


def _jitter(image: Image.Image, amount: float, seed: int) -> Image.Image:
    random_generator = random.Random(seed)
    factors = [random_generator.uniform(1.0 - amount, 1.0 + amount) for _ in range(3)]
    transformed = ImageEnhance.Brightness(image).enhance(factors[0])
    transformed = ImageEnhance.Contrast(transformed).enhance(factors[1])
    return ImageEnhance.Color(transformed).enhance(factors[2])


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * ratio))
    crop_height = max(1, round(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def apply_transform(
    image: Image.Image, spec: TransformSpec, seed: int = 0
) -> Image.Image:
    image = image.convert("RGB")
    if spec.family == "jpeg":
        transformed = _jpeg(image, round(spec.value))
    elif spec.family == "blur":
        transformed = image.filter(ImageFilter.GaussianBlur(radius=spec.value))
    elif spec.family == "resize":
        transformed = _resize(image, spec.value)
    elif spec.family == "noise":
        transformed = _noise(image, spec.value, seed)
    elif spec.family == "jitter":
        transformed = _jitter(image, spec.value, seed)
    elif spec.family == "crop":
        transformed = _center_crop(image, spec.value)
    else:
        raise ValueError(f"Unknown transform family: {spec.family}")
    return transformed.convert("RGB")


def choose_training_transform(global_seed: int, image_id: str) -> TransformSpec:
    families = sorted({spec.family for spec in EVAL_TRANSFORMS})
    family_seed = stable_seed(global_seed, image_id, "family")
    family = random.Random(family_seed).choice(families)
    candidates = [spec for spec in EVAL_TRANSFORMS if spec.family == family]
    severity_seed = stable_seed(global_seed, image_id, family)
    return random.Random(severity_seed).choice(candidates)


def apply_training_transform(
    image: Image.Image, global_seed: int, image_id: str
) -> tuple[Image.Image, TransformSpec]:
    spec = choose_training_transform(global_seed, image_id)
    seed = stable_seed(global_seed, image_id, spec.key)
    return apply_transform(image, spec, seed=seed), spec
