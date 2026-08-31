from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.detector.features import FeatureArrays, load_feature_cache
from src.detector.model import (
    BinaryMetrics,
    calculate_metrics,
    load_artifact,
    predict_scores,
    resolve_feature_mode,
)
from src.detector.transforms import EVAL_TRANSFORMS


def condition_family(condition: str) -> str:
    return "clean" if condition == "clean" else condition.split("_", maxsplit=1)[0]


def evaluate_models(
    models: dict,
    arrays: FeatureArrays,
    threshold: float,
    semantic_models: set[str] | None = None,
) -> dict[str, BinaryMetrics]:
    semantic_models = {"semantic_clean"} if semantic_models is None else semantic_models
    return {
        name: calculate_metrics(
            arrays.labels,
            predict_scores(model, arrays, semantic_only=name in semantic_models),
            threshold,
        )
        for name, model in models.items()
    }


def aggregate_auc(
    condition_metrics: dict[str, BinaryMetrics], clean_condition: str = "clean"
) -> dict[str, float | dict[str, float]]:
    if clean_condition not in condition_metrics:
        raise ValueError("Clean metrics are required for composite scoring")
    transformed = {
        condition: result
        for condition, result in condition_metrics.items()
        if condition != clean_condition
    }
    if not transformed:
        raise ValueError("At least one transformed condition is required")

    family_values: dict[str, list[float]] = defaultdict(list)
    for condition, result in transformed.items():
        family_values[condition_family(condition)].append(result.auc)
    family_auc = {
        family: float(np.mean(values))
        for family, values in sorted(family_values.items())
    }
    clean_auc = condition_metrics[clean_condition].auc
    condition_weighted = float(np.mean([result.auc for result in transformed.values()]))
    family_balanced = float(np.mean(list(family_auc.values())))
    return {
        "clean_auc": clean_auc,
        "condition_weighted_robust_auc": condition_weighted,
        "family_balanced_robust_auc": family_balanced,
        "condition_weighted_final_score": 0.5 * (clean_auc + condition_weighted),
        "family_balanced_final_score": 0.5 * (clean_auc + family_balanced),
        "family_auc": family_auc,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score detector caches and robustness."
    )
    parser.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", *(spec.key for spec in EVAL_TRANSFORMS)],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "clean" not in args.conditions:
        raise ValueError("--conditions must include clean")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("--conditions contains duplicates")

    artifact = load_artifact(args.model)
    models = artifact["models"]
    threshold = artifact["config"]["threshold"]
    semantic_models = {"semantic_clean"}
    if resolve_feature_mode(artifact["config"]) == "semantic":
        semantic_models.add(artifact["final_model"])
    results: dict[str, dict[str, BinaryMetrics]] = {}
    counts: dict[str, int] = {}
    for condition in args.conditions:
        arrays = load_feature_cache(args.features_root / f"test-{condition}")
        results[condition] = evaluate_models(models, arrays, threshold, semantic_models)
        counts[condition] = len(arrays)
        print(f"scored condition={condition} rows={len(arrays)}")

    final_model = artifact["final_model"]
    robustness_rows = []
    for condition in args.conditions:
        metrics = results[condition][final_model]
        severity = (
            "none" if condition == "clean" else condition.split("_", maxsplit=1)[1]
        )
        robustness_rows.append(
            {
                "model": final_model,
                "condition": condition,
                "family": condition_family(condition),
                "severity": severity,
                "count": counts[condition],
                **asdict(metrics),
                "threshold": threshold,
            }
        )
    write_csv(robustness_rows, args.output_dir / "robustness_table.csv")

    ablation_rows = []
    summaries = {}
    for model_name in models:
        model_conditions = {
            condition: condition_results[model_name]
            for condition, condition_results in results.items()
        }
        summary = aggregate_auc(model_conditions)
        summaries[model_name] = summary
        ablation_rows.append(
            {
                "model": model_name,
                "clean_auc": summary["clean_auc"],
                "condition_weighted_robust_auc": summary[
                    "condition_weighted_robust_auc"
                ],
                "family_balanced_robust_auc": summary["family_balanced_robust_auc"],
                "condition_weighted_final_score": summary[
                    "condition_weighted_final_score"
                ],
                "family_balanced_final_score": summary["family_balanced_final_score"],
                "conditions_evaluated": len(args.conditions) - 1,
                "families_evaluated": len(summary["family_auc"]),
                "threshold": threshold,
            }
        )
    write_csv(ablation_rows, args.output_dir / "ablation_table.csv")

    summary_payload = {
        "aggregation": {
            "headline": "mean AUC across transformed conditions",
            "sensitivity": "mean within family, then mean across families",
            "final_score": "0.5 * clean_auc + 0.5 * robust_auc",
        },
        "conditions": args.conditions,
        "final_model": final_model,
        "models": summaries,
    }
    (args.output_dir / "robustness_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    final_summary = summaries[final_model]
    print(
        f"final_model={final_model} clean_auc={final_summary['clean_auc']:.6f} "
        f"robust_auc={final_summary['condition_weighted_robust_auc']:.6f} "
        f"final_score={final_summary['condition_weighted_final_score']:.6f}"
    )


if __name__ == "__main__":
    main()
