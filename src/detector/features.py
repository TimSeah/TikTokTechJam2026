from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.detector.data import ImageRecord
from src.detector.freq_features import radial_fft_features
from src.detector.transforms import (
    EVAL_TRANSFORMS,
    TransformSpec,
    apply_training_transform,
    apply_transform,
    stable_seed,
)

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_CHECKPOINT = "openai"
FFT_BINS = 32
FFT_IMAGE_SIZE = 256
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureArrays:
    semantic: np.ndarray
    frequency: np.ndarray
    labels: np.ndarray
    image_ids: np.ndarray
    relative_paths: np.ndarray
    transform_keys: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested, but no GPU is available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_backbone(
    device: torch.device,
    model_name: str = MODEL_NAME,
    pretrained_checkpoint: str = PRETRAINED_CHECKPOINT,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_checkpoint
    )
    return model.eval().to(device), preprocess


def manifest_sha256(manifest_path: Path) -> str:
    digest = hashlib.sha256()
    with manifest_path.open("rb") as manifest_file:
        for chunk in iter(lambda: manifest_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_condition(condition: str) -> str | TransformSpec:
    if condition in {"clean", "augmented"}:
        return condition
    by_key = {spec.key: spec for spec in EVAL_TRANSFORMS}
    if condition not in by_key:
        valid = ", ".join(["clean", "augmented", *by_key])
        raise ValueError(f"Unknown condition {condition!r}. Expected one of: {valid}")
    return by_key[condition]


class FeatureDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        data_root: Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        condition: str | TransformSpec,
        seed: int,
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.preprocess = preprocess
        self.condition = condition
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_root / record.relative_path
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        if self.condition == "clean":
            transformed = image
            transform_key = "clean"
        elif self.condition == "augmented":
            transformed, spec = apply_training_transform(
                image, self.seed, record.image_id
            )
            transform_key = spec.key
        elif isinstance(self.condition, TransformSpec):
            transform_seed = stable_seed(self.seed, record.image_id, self.condition.key)
            transformed = apply_transform(image, self.condition, seed=transform_seed)
            transform_key = self.condition.key
        else:
            raise ValueError(f"Unsupported condition: {self.condition}")

        semantic_input = self.preprocess(transformed)
        frequency = radial_fft_features(
            transformed, bins=FFT_BINS, image_size=FFT_IMAGE_SIZE
        )
        return (
            semantic_input,
            frequency,
            record.label,
            record.image_id,
            record.relative_path,
            transform_key,
        )


def create_loader(
    dataset: FeatureDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def encode_loader(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> FeatureArrays:
    semantic_rows: list[np.ndarray] = []
    frequency_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    image_ids: list[str] = []
    relative_paths: list[str] = []
    transform_keys: list[str] = []

    for image_tensors, frequencies, labels, ids, paths, keys in loader:
        image_tensors = image_tensors.to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    semantic = model.encode_image(image_tensors)
            else:
                semantic = model.encode_image(image_tensors)
            semantic = functional.normalize(semantic.float(), dim=-1)

        semantic_rows.append(semantic.cpu().numpy().astype(np.float16))
        frequency_rows.append(frequencies.numpy().astype(np.float32, copy=False))
        label_rows.append(labels.numpy().astype(np.int8, copy=False))
        image_ids.extend(ids)
        relative_paths.extend(paths)
        transform_keys.extend(keys)

    return FeatureArrays(
        semantic=np.concatenate(semantic_rows),
        frequency=np.concatenate(frequency_rows),
        labels=np.concatenate(label_rows),
        image_ids=np.asarray(image_ids),
        relative_paths=np.asarray(relative_paths),
        transform_keys=np.asarray(transform_keys),
    )


def save_feature_shard(arrays: FeatureArrays, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("wb") as output_file:
        np.savez(
            output_file,
            semantic=arrays.semantic,
            frequency=arrays.frequency,
            labels=arrays.labels,
            image_ids=arrays.image_ids,
            relative_paths=arrays.relative_paths,
            transform_keys=arrays.transform_keys,
        )
    temporary_path.replace(output_path)


def load_feature_cache(cache_dir: Path) -> FeatureArrays:
    shard_paths = sorted(cache_dir.glob("part-*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found in {cache_dir}")

    rows: dict[str, list[np.ndarray]] = {
        "semantic": [],
        "frequency": [],
        "labels": [],
        "image_ids": [],
        "relative_paths": [],
        "transform_keys": [],
    }
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            for key in rows:
                rows[key].append(shard[key])
    return FeatureArrays(
        **{key: np.concatenate(values) for key, values in rows.items()}
    )


def write_cache_metadata(cache_dir: Path, metadata: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    temporary_path = metadata_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)


def read_cache_metadata(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
