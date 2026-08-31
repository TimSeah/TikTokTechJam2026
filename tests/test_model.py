from pathlib import Path

import numpy as np

from src.detector.features import FeatureArrays
from src.detector.model import (
    ARTIFACT_SCHEMA_VERSION,
    FINAL_MODEL_NAME,
    load_artifact,
    predict_margins,
    predict_scores,
    save_artifact,
    score_models,
    train_models,
)


def _arrays(offset: float = 0.0) -> FeatureArrays:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    semantic = np.asarray(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    frequency = semantic[:, :1] + offset
    return FeatureArrays(
        semantic=semantic,
        frequency=frequency,
        labels=labels,
        image_ids=np.asarray(["a", "b", "c", "d"]),
        relative_paths=np.asarray(["a", "b", "c", "d"]),
        transform_keys=np.asarray(["clean"] * 4),
    )


def test_three_models_train_and_score() -> None:
    models = train_models(_arrays(), _arrays(0.1), seed=7)
    metrics = score_models(models, _arrays())
    assert set(models) == {"semantic_clean", "hybrid_clean", "hybrid_augmented"}
    assert all(result.auc == 1.0 for result in metrics.values())


def test_artifact_round_trip(tmp_path: Path) -> None:
    models = train_models(_arrays(), _arrays(0.1), seed=7)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "final_model": FINAL_MODEL_NAME,
        "models": models,
        "config": {"semantic_dimension": 2, "frequency_dimension": 1},
    }
    output_path = tmp_path / "model.joblib"
    expected = predict_scores(models[FINAL_MODEL_NAME], _arrays(), semantic_only=False)
    save_artifact(artifact, output_path)
    loaded = load_artifact(output_path)
    actual = predict_scores(
        loaded["models"][FINAL_MODEL_NAME], _arrays(), semantic_only=False
    )
    np.testing.assert_allclose(actual, expected)


def test_decision_margins_preserve_ranking_after_probabilities_saturate() -> None:
    model = train_models(_arrays(), _arrays(0.1), seed=7)[FINAL_MODEL_NAME]
    classifier = model.named_steps["classifier"]
    classifier.intercept_ += 1000.0

    scores = predict_scores(model, _arrays(), semantic_only=False)
    margins = predict_margins(model, _arrays(), semantic_only=False)

    assert np.count_nonzero(scores == 1.0) == len(scores)
    assert len(np.unique(margins)) == len(margins)
