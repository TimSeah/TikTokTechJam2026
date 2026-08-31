from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SID_REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
WILDFAKE_TRAIN_REVISION = "3c4a1d3824e593167f9b6f682c079c3c17516214"
WILDFAKE_EVAL_REVISION = "a24ae914e4fe8ae1956c671b4fcd902f5bca1a0d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract semantic caches and train the gated SID/WildFake detector "
            "under a hard wall-clock budget."
        )
    )
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--timing-out", type=Path, default=Path("outputs/native_training_timing.json")
    )
    return parser.parse_args()


def embed_args(
    manifest: str,
    data_root: str,
    output_dir: str,
    condition: str,
    dataset: str,
    revision: str,
    source_split: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        "-m",
        "src.detector.embed",
        "--manifest",
        manifest,
        "--data-root",
        data_root,
        "--output-dir",
        output_dir,
        "--condition",
        condition,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--shard-size",
        "2000",
        "--seed",
        str(args.seed),
        "--semantic-only",
        "--dataset",
        dataset,
        "--dataset-revision",
        revision,
        "--source-split",
        source_split,
    ]


def build_stages(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    sid_common = (
        "saberzl/SID_Set",
        SID_REVISION,
    )
    wildfake_train_common = (
        "buxtcodes/WildFake-Sample",
        WILDFAKE_TRAIN_REVISION,
    )
    wildfake_eval_common = (
        "techjam-aigc/wildfake-eval-subset",
        WILDFAKE_EVAL_REVISION,
    )
    stages = [
        (
            "sid_train_clean",
            embed_args(
                "data/native-train/sid-final/manifest-train.csv",
                "data/native-train/sid-final",
                "data/features/sid-final-clean",
                "clean",
                *sid_common,
                "train",
                args,
            ),
        ),
        (
            "sid_train_augmented",
            embed_args(
                "data/native-train/sid-final/manifest-train.csv",
                "data/native-train/sid-final",
                "data/features/sid-final-augmented",
                "augmented",
                *sid_common,
                "train",
                args,
            ),
        ),
        (
            "wildfake_train_clean",
            embed_args(
                "data/native-train/wildfake-final/manifest-train.csv",
                "data/native-train/wildfake-final",
                "data/features/wildfake-final-clean",
                "clean",
                *wildfake_train_common,
                "train",
                args,
            ),
        ),
        (
            "wildfake_train_augmented",
            embed_args(
                "data/native-train/wildfake-final/manifest-train.csv",
                "data/native-train/wildfake-final",
                "data/features/wildfake-final-augmented",
                "augmented",
                *wildfake_train_common,
                "train",
                args,
            ),
        ),
        (
            "sid_calibration",
            embed_args(
                "data/native-train/sid-final/manifest-eval.csv",
                "data/native-train/sid-final",
                "data/features/sid-final-calibration",
                "clean",
                *sid_common,
                "train",
                args,
            ),
        ),
        (
            "sid_validation",
            embed_args(
                "data/manifests/native-eval/sid-validation.csv",
                "data/blind-test",
                "data/features/sid-validation-clean",
                "clean",
                *sid_common,
                "validation",
                args,
            ),
        ),
        (
            "wildfake_default",
            embed_args(
                "data/manifests/native-eval/wildfake-default.csv",
                "data/blind-test",
                "data/features/wildfake-default-clean",
                "clean",
                *wildfake_eval_common,
                "validation",
                args,
            ),
        ),
        (
            "wildfake_matched",
            embed_args(
                "data/manifests/native-eval/wildfake-laion-matched.csv",
                "data/blind-test",
                "data/features/wildfake-laion-matched-clean",
                "clean",
                *wildfake_eval_common,
                "validation",
                args,
            ),
        ),
    ]
    stages.append(
        (
            "fit_and_gate",
            [
                "-m",
                "src.detector.train_native_probe",
                "--cifake-train-cache",
                "data/features/train-clean",
                "--cifake-augmented-train-cache",
                "data/features/train-augmented",
                "--native-train-cache",
                "data/features/sid-final-clean",
                "--native-augmented-train-cache",
                "data/features/sid-final-augmented",
                "--wildfake-train-cache",
                "data/features/wildfake-final-clean",
                "--wildfake-augmented-train-cache",
                "data/features/wildfake-final-augmented",
                "--calibration-cache",
                "data/features/sid-final-calibration",
                "--cifake-eval-cache",
                "data/features/test-clean",
                "--sid-eval-cache",
                "data/features/sid-validation-clean",
                "--wildfake-eval-cache",
                "data/features/wildfake-default-clean",
                "--wildfake-matched-eval-cache",
                "data/features/wildfake-laion-matched-clean",
                "--output",
                "outputs/model-native.joblib",
                "--metrics-out",
                "outputs/native_metrics.json",
                "--promote-to",
                "outputs/model.joblib",
                "--seed",
                str(args.seed),
            ],
        )
    )
    return stages


def write_timing(
    destination: Path,
    budget_seconds: float,
    started: float,
    status: str,
    stages: list[dict],
) -> None:
    payload = {
        "budget_seconds": budget_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": status,
        "stages": stages,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(destination)


def main() -> None:
    args = parse_args()
    if args.max_minutes <= 0 or args.batch_size < 1 or args.workers < 0:
        raise ValueError(
            "Budget and batch size must be positive; workers cannot be negative"
        )
    budget_seconds = args.max_minutes * 60
    started = time.monotonic()
    results: list[dict] = []
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    write_timing(args.timing_out, budget_seconds, started, "running", results)
    for name, command_args in build_stages(args):
        remaining = budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            write_timing(
                args.timing_out, budget_seconds, started, "budget_exhausted", results
            )
            raise RuntimeError(f"Training budget exhausted before {name}")
        print(f"stage={name} remaining_seconds={remaining:.1f}", flush=True)
        stage_started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, *command_args],
                check=False,
                timeout=remaining,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "name": name,
                    "seconds": round(time.monotonic() - stage_started, 3),
                    "status": "timed_out",
                }
            )
            write_timing(
                args.timing_out, budget_seconds, started, "budget_exhausted", results
            )
            raise RuntimeError(f"Training budget exhausted during {name}") from exc
        results.append(
            {
                "name": name,
                "seconds": round(time.monotonic() - stage_started, 3),
                "status": "passed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            write_timing(args.timing_out, budget_seconds, started, "failed", results)
            raise RuntimeError(f"Stage {name} exited with {completed.returncode}")
        write_timing(args.timing_out, budget_seconds, started, "running", results)
    write_timing(args.timing_out, budget_seconds, started, "completed", results)
    print(f"training_pipeline_seconds={time.monotonic() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
