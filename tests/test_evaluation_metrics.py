import numpy as np

from src.model.training import compute_metrics


def test_macro_metrics_ignore_classes_absent_from_true_labels():
    y = np.array([6] * 10 + [7] * 10)
    preds = np.array([6] * 9 + [0] + [7] * 8 + [2, 3])

    metrics = compute_metrics(y, preds, loss=0.1)

    assert metrics["accuracy"] == 0.85
    assert metrics["recall_macro"] == (0.9 + 0.8) / 2
    assert metrics["f1_macro"] > 0.5


def test_macro_metrics_match_full_label_set_when_all_classes_present():
    y = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 1, 1, 1, 2, 0])

    metrics = compute_metrics(y, preds, loss=0.2)
    assert metrics["accuracy"] == 4 / 6
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_empty_predictions_not_required_but_labels_still_only_true_classes():
    y = np.array([6, 6, 7, 7])
    preds = np.array([0, 0, 7, 7])

    metrics = compute_metrics(y, preds, loss=0.0)
    assert metrics["recall_macro"] == (0.0 + 1.0) / 2
