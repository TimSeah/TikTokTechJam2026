import numpy as np

from src.detector.analyze_errors import select_errors


def test_select_errors_ranks_most_confident_mistakes() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.2, 0.8, 0.9, 0.1, 0.3, 0.8])
    examples = select_errors(labels, scores, per_type=2, threshold=0.5)
    assert [(example.error_type, example.index) for example in examples] == [
        ("false_positive", 2),
        ("false_positive", 1),
        ("false_negative", 3),
        ("false_negative", 4),
    ]
