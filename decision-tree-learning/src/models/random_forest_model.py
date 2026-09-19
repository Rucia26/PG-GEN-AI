"""
A thin, learning-friendly wrapper around sklearn's RandomForestClassifier.

Mirrors the interface of DecisionTreeModel (fit/predict/predict_proba/
evaluate) on purpose: notebook 07 compares tree vs. forest vs. boosting by
calling the exact same methods on each, so the comparison code doesn't need
to know which model it's holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import SETTINGS
from src.evaluation.model_evaluator import EvaluationMetrics, evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel:
    """
    Wraps sklearn.ensemble.RandomForestClassifier for the churn classification task.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Each tree is trained on a different
        bootstrap sample (bagging) and considers a random subset of features
        at each split — see notebook 07 for why both of these matter.
    max_depth : int | None
        Maximum depth per tree. Individual trees in a forest are often
        allowed to grow deeper than a standalone tree would, since averaging
        many overfit trees reduces variance (the core idea of bagging).
    random_state : int | None
        Defaults to the project-wide seed in config.yaml.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state if random_state is not None else SETTINGS.model.random_state,
        )
        logger.info(
            "Initialized RandomForestModel(n_estimators=%s, max_depth=%s)",
            n_estimators, max_depth,
        )

    def fit(self, training_features: pd.DataFrame, training_target: pd.Series) -> "RandomForestModel":
        """Train the forest on the given features and target."""
        self.classifier.fit(training_features, training_target)
        logger.info("Model trained on %d samples, %d features", *training_features.shape)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0/1), by majority vote across trees."""
        return self.classifier.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities, averaged across all trees."""
        return self.classifier.predict_proba(features)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> EvaluationMetrics:
        """Predict on the given features and compute standard classification metrics."""
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)[:, 1]
        return evaluate_predictions(target, predictions, probabilities)

    @property
    def feature_importances(self) -> np.ndarray:
        """Feature importances averaged across every tree in the forest."""
        return self.classifier.feature_importances_
