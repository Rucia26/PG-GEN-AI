"""
Compute and report standard classification metrics.

Kept separate from the model classes in src/models/ so that ANY model
(decision tree, random forest, gradient boosting) can be evaluated the same
way — evaluation logic shouldn't be duplicated per model type.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class EvaluationMetrics:
    """Container for the standard classification metrics we track."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> dict[str, float]:
        """Return the scalar metrics (excludes the confusion matrix) as a dict."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def evaluate_predictions(
    actual_target: np.ndarray,
    predicted_target: np.ndarray,
    predicted_probabilities: np.ndarray | None = None,
) -> EvaluationMetrics:
    """
    Compute accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

    Parameters
    ----------
    actual_target : array-like
        Ground-truth labels (0/1).
    predicted_target : array-like
        Model's predicted labels (0/1).
    predicted_probabilities : array-like, optional
        Predicted probability of the positive class (churn=1). Required for
        ROC-AUC; if omitted, roc_auc is set to NaN rather than raising, since
        not every caller has probabilities on hand.

    Returns
    -------
    EvaluationMetrics
        All metrics needed for the model comparison table (notebook 05+).

    Notes
    -----
    precision/recall/f1 use average="binary" (the default) with the positive
    label being 1 (churn) — this matches how the business cares about this
    problem: correctly catching customers who WILL churn.
    """
    roc_auc = (
        roc_auc_score(actual_target, predicted_probabilities)
        if predicted_probabilities is not None
        else float("nan")
    )

    metrics = EvaluationMetrics(
        accuracy=accuracy_score(actual_target, predicted_target),
        precision=precision_score(actual_target, predicted_target, zero_division=0),
        recall=recall_score(actual_target, predicted_target, zero_division=0),
        f1=f1_score(actual_target, predicted_target, zero_division=0),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(actual_target, predicted_target),
    )

    logger.info(
        "Evaluation - accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
        metrics.accuracy, metrics.precision, metrics.recall, metrics.f1, metrics.roc_auc,
    )

    return metrics
