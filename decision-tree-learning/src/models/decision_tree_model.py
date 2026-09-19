"""
A thin, learning-friendly wrapper around sklearn's DecisionTreeClassifier.

WHY WRAP IT AT ALL, IF SKLEARN ALREADY HAS fit/predict?
Two reasons specific to this learning project:
1. `evaluate()` bundles prediction + metrics into one call, reused
   identically by random_forest_model.py and gradient_boosting_model.py —
   so notebook 05+ can compare models without repeating evaluation code.
2. Construction is logged, so you can see in the console/log file exactly
   which hyperparameters were used for a given run — useful once you start
   the overfitting experiments in notebook 04 with many different
   max_depth values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.config.settings import SETTINGS
from src.evaluation.model_evaluator import EvaluationMetrics, evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionTreeModel:
    """
    Wraps sklearn.tree.DecisionTreeClassifier for the churn classification task.

    Parameters
    ----------
    criterion : str
        Split quality measure: "gini" (default, faster) or "entropy"
        (information gain). See notebook 04 for what these mean and how
        they differ in practice.
    max_depth : int | None
        Maximum tree depth. None means nodes expand until pure or until
        min_samples_split is reached — this is how a tree overfits (see the
        max_depth experiments in notebook 04).
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    random_state : int | None
        Controls the randomness sklearn uses when multiple splits tie on
        quality. Defaults to the project-wide seed in config.yaml so results
        are reproducible run-to-run.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.classifier = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state if random_state is not None else SETTINGS.model.random_state,
        )
        logger.info(
            "Initialized DecisionTreeModel(criterion=%s, max_depth=%s, "
            "min_samples_split=%s, min_samples_leaf=%s)",
            criterion, max_depth, min_samples_split, min_samples_leaf,
        )

    def fit(self, training_features: pd.DataFrame, training_target: pd.Series) -> "DecisionTreeModel":
        """
        Train the decision tree on the given features and target.

        Parameters
        ----------
        training_features : pd.DataFrame
            Encoded, numeric training features (output of preprocessor.py).
        training_target : pd.Series
            Training labels (0/1).

        Returns
        -------
        DecisionTreeModel
            self, so calls can be chained: model.fit(X, y).evaluate(...)
        """
        self.classifier.fit(training_features, training_target)
        logger.info("Model trained on %d samples, %d features", *training_features.shape)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0/1) for the given features."""
        return self.classifier.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return predicted class probabilities for the given features.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2): column 0 is P(no churn), column 1 is
            P(churn). Most evaluation code below uses column 1.
        """
        return self.classifier.predict_proba(features)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> EvaluationMetrics:
        """
        Predict on the given features and compute standard classification metrics.

        Parameters
        ----------
        features : pd.DataFrame
            Features to evaluate on (typically the held-out test set).
        target : pd.Series
            True labels corresponding to `features`.

        Returns
        -------
        EvaluationMetrics
            accuracy, precision, recall, f1, roc_auc, confusion_matrix.
        """
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)[:, 1]
        return evaluate_predictions(target, predictions, probabilities)

    @property
    def feature_importances(self) -> np.ndarray:
        """Feature importance scores from the fitted tree (requires fit() first)."""
        return self.classifier.feature_importances_
