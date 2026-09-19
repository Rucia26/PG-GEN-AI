"""
Tests for src/evaluation/model_evaluator.py.
"""

import numpy as np

from src.evaluation.model_evaluator import evaluate_predictions


def test_perfect_predictions_give_perfect_metrics():
    """When predictions exactly match the truth, every metric should be 1.0."""
    actual = np.array([0, 1, 1, 0, 1])
    predicted = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.9, 0.1, 0.9])

    metrics = evaluate_predictions(actual, predicted, probabilities)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_always_predicting_majority_class_has_zero_recall():
    """
    Mirrors notebook 05's baseline demonstration: a model that always
    predicts the majority class (0) catches zero positives, so recall
    must be exactly 0.
    """
    actual = np.array([0, 0, 0, 1, 1])
    predicted = np.array([0, 0, 0, 0, 0])

    metrics = evaluate_predictions(actual, predicted)

    assert metrics.recall == 0.0
    assert metrics.accuracy == 0.6  # 3 correct out of 5


def test_confusion_matrix_has_correct_shape_for_binary_classification():
    """The confusion matrix should always be 2x2 for a binary problem."""
    actual = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 1, 0])

    metrics = evaluate_predictions(actual, predicted)

    assert metrics.confusion_matrix.shape == (2, 2)
    assert metrics.confusion_matrix.sum() == len(actual)


def test_roc_auc_is_nan_when_probabilities_not_provided():
    """roc_auc should gracefully become NaN, not raise, when probabilities are omitted."""
    actual = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 0, 1])

    metrics = evaluate_predictions(actual, predicted, predicted_probabilities=None)

    assert np.isnan(metrics.roc_auc)


def test_to_dict_excludes_confusion_matrix():
    """to_dict() should return only scalar metrics, not the confusion matrix array."""
    actual = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 0, 1])

    metrics = evaluate_predictions(actual, predicted)
    metrics_dict = metrics.to_dict()

    assert "confusion_matrix" not in metrics_dict
    assert set(metrics_dict.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
