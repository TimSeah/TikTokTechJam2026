from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.detector.data import read_manifest
from src.detector.features import (
    FeatureDataset,
    create_loader,
    encode_loader,
    load_backbone,
    load_feature_cache,
    resolve_device,
)
from src.detector.model import load_artifact, predict_margins, predict_scores
from src.detector.shift_audit import (
    reliability_diagnostics,
    stratified_bootstrap_auc,
    threshold_diagnostics,
    write_json_atomic,
)
from src.detector.transforms import EVAL_TRANSFORMS, PLATFORM_STYLE_CHAINS

DATASETS = {
    "sid_validation": {
        "manifest": "data/manifests/native-eval/sid-validation.csv",
        "clean_cache": "data/features/sid-validation-clean",
    },
    "wildfake_coco_dalle": {
        "manifest": "data/manifests/native-eval/wildfake-default.csv",
        "clean_cache": "data/features/wildfake-default-clean",
    },
    "wildfake_laion_dalle": {
        "manifest": "data/manifests/native-eval/wildfake-laion-matched.csv",
        "clean_cache": "data/features/wildfake-laion-matched-clean",
    },
}

CONDITIONS = {
    "clean": "clean",
    **{spec.key: spec for spec in EVAL_TRANSFORMS},
    **{chain.key: chain for chain in PLATFORM_STYLE_CHAINS},
}


def condition_kind(key: str) -> str:
    if key == "clean":
        return "clean"
    if key.startswith("chain_"):
        return "platform_style_chain"
    return "individual_transform"


def summarize_conditions(results: dict[str, dict]) -> dict:
    summary = {}
    for kind in ("individual_transform", "platform_style_chain"):
        aucs = [
            result["probability_ranking"]["auc"]
            for result in results.values()
            if result["kind"] == kind
        ]
        if aucs:
            summary[kind] = {
                "condition_count": len(aucs),
                "mean_auc": float(np.mean(aucs)),
                "minimum_auc": min(aucs),
                "maximum_auc": max(aucs),
            }
    clean_auc = results["clean"]["probability_ranking"]["auc"]
    if "individual_transform" in summary:
        summary["individual_transform"]["composite_with_clean"] = float(
            0.5 * (clean_auc + summary["individual_transform"]["mean_auc"])
        )
    return summary


def evaluate_condition(model, arrays, threshold: float, key: str) -> dict:
    probabilities = predict_scores(model, arrays, semantic_only=True)
    margins = predict_margins(model, arrays, semantic_only=True)
    return {
        "kind": condition_kind(key),
        "count": len(arrays),
        "probability_ranking": stratified_bootstrap_auc(
            arrays.labels, probabilities, replicates=1000
        ),
        "margin_ranking": stratified_bootstrap_auc(
            arrays.labels, margins, replicates=1000
        ),
        "probability_saturation": {
            "exact_zero": int(np.count_nonzero(probabilities == 0.0)),
            "exact_one": int(np.count_nonzero(probabilities == 1.0)),
            "unique_probabilities": int(len(np.unique(probabilities))),
        },
        "threshold_metrics": threshold_diagnostics(
            arrays.labels, probabilities, threshold
        ),
        "calibration": reliability_diagnostics(arrays.labels, probabilities),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the promoted model on native-resolution transform stresses"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/native_stress.json")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    artifact = load_artifact(root / args.model)
    model_sha256 = hashlib.sha256((root / args.model).read_bytes()).hexdigest()
    model = artifact["models"][artifact["final_model"]]
    threshold = float(artifact["config"]["threshold"])
    transformed_requested = any(key != "clean" for key in args.conditions)
    device = resolve_device(args.device)
    backbone = preprocess = None
    if transformed_requested:
        backbone, preprocess = load_backbone(device)

    datasets = {}
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        records = read_manifest(root / spec["manifest"])
        results = {}
        for key in args.conditions:
            if key == "clean":
                arrays = load_feature_cache(root / spec["clean_cache"])
            else:
                if backbone is None or preprocess is None:
                    raise RuntimeError(
                        "Backbone was not loaded for transformed evaluation"
                    )
                dataset = FeatureDataset(
                    records,
                    root / "data/blind-test",
                    preprocess,
                    CONDITIONS[key],
                    seed=args.seed,
                    semantic_only=True,
                )
                arrays = encode_loader(
                    backbone,
                    create_loader(dataset, args.batch_size, args.workers, device),
                    device,
                )
            results[key] = evaluate_condition(model, arrays, threshold, key)
            print(
                f"dataset={dataset_name} condition={key} "
                f"auc={results[key]['probability_ranking']['auc']:.6f}",
                flush=True,
            )
        datasets[dataset_name] = {
            "conditions": results,
            "summary": summarize_conditions(results),
        }

    payload = {
        "schema_version": 1,
        "protocol": {
            "model_artifact": str(args.model).replace("\\", "/"),
            "model_sha256": model_sha256,
            "threshold": threshold,
            "seed": args.seed,
            "individual_transform_count": len(EVAL_TRANSFORMS),
            "chains": {
                chain.key: [spec.key for spec in chain.steps]
                for chain in PLATFORM_STYLE_CHAINS
            },
            "chain_scope": (
                "Heuristic platform-style stress tests, not measurements of a "
                "proprietary TikTok processing pipeline."
            ),
            "auc_interval": "1000-replicate stratified percentile bootstrap",
        },
        "datasets": datasets,
    }
    output_path = args.output if args.output.is_absolute() else root / args.output
    write_json_atomic(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
