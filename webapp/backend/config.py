from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(WEBAPP_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WEBAPP_ROOT / path
    return path.resolve()


def _path_from_env(name: str, default: str) -> Path:
    return _resolve_path(os.getenv(name, default))


def _paths_from_env(name: str) -> tuple[Path, ...]:
    return tuple(
        _resolve_path(value.strip())
        for value in os.getenv(name, "").split(",")
        if value.strip()
    )


@dataclass(frozen=True)
class Settings:
    model_artifact_path: Path
    model_device: str
    challenge_dataset_path: Path
    allowed_origins: tuple[str, ...]
    round_cache_size: int
    challenge_dataset_paths: tuple[Path, ...] = ()

    @property
    def challenge_dataset_roots(self) -> tuple[Path, ...]:
        return self.challenge_dataset_paths or (self.challenge_dataset_path,)

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(
                ","
            )
            if origin.strip()
        )
        dataset_paths = _paths_from_env("CHALLENGE_DATASET_PATHS")
        if not dataset_paths:
            dataset_paths = (
                _path_from_env(
                    "CHALLENGE_DATASET_PATH", "../data/downloads/test"
                ),
            )
        return cls(
            model_artifact_path=_path_from_env(
                "MODEL_ARTIFACT_PATH", "../outputs/model.joblib"
            ),
            model_device=os.getenv("MODEL_DEVICE", "auto").strip().lower(),
            challenge_dataset_path=dataset_paths[0],
            allowed_origins=origins,
            round_cache_size=max(10, int(os.getenv("ROUND_CACHE_SIZE", "500"))),
            challenge_dataset_paths=dataset_paths,
        )
