from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(WEBAPP_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    if not path.is_absolute():
        path = WEBAPP_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    model_artifact_path: Path
    model_device: str
    challenge_dataset_path: Path
    allowed_origins: tuple[str, ...]
    round_cache_size: int

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(
                ","
            )
            if origin.strip()
        )
        return cls(
            model_artifact_path=_path_from_env(
                "MODEL_ARTIFACT_PATH", "../outputs/development-model.joblib"
            ),
            model_device=os.getenv("MODEL_DEVICE", "auto").strip().lower(),
            challenge_dataset_path=_path_from_env(
                "CHALLENGE_DATASET_PATH", "../data/downloads/test"
            ),
            allowed_origins=origins,
            round_cache_size=max(10, int(os.getenv("ROUND_CACHE_SIZE", "500"))),
        )
