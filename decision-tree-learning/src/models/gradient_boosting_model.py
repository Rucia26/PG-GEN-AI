"""
A thin, learning-friendly wrapper around sklearn's GradientBoostingClassifier.

Same fit/predict/predict_proba/evaluate interface as DecisionTreeModel and
RandomForestModel, so notebook 07 can compare all three models with
identical calling code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.config.settings import SETTINGS
from src.evaluation.model_evaluator import EvaluationMetrics, evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradientBoostingModel:
    """
    Wraps sklearn.ensemble.GradientBoostingClassifier for the churn classification task.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (trees added sequentially). Unlike a
        random forest, more estimators here means more ROUNDS of
        error-correction, not more independent voters — see notebook 07.
    learning_rate : float
        Shrinks the contribution of each new tree. Lower values need more
        estimators to reach the same fit but usually generalize better —
        the classic boosting speed/robustness trade-off.
    max_depth : int
        Depth of each individual boosting tree. Boosting typically uses
        much shallower trees ("weak learners") than bagging does, since
        the ensemble — not any single tree — is meant to capture complexity.
    random_state : int | None
        Defaults to the project-wide seed in config.yaml.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state if random_state is not None else SETTINGS.model.random_state,
        )
        logger.info(
            "Initialized GradientBoostingModel(n_estimators=%s, learning_rate=%s, max_depth=%s)",
            n_estimators, learning_rate, max_depth,
        )

    def fit(self, training_features: pd.DataFrame, training_target: pd.Series) -> "GradientBoostingModel":
        """Train the boosted ensemble sequentially on the given features and target."""
        self.classifier.fit(training_features, training_target)
        logger.info("Model trained on %d samples, %d features", *training_features.shape)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0/1) from the final boosted ensemble."""
        return self.classifier.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities from the final boosted ensemble."""
        return self.classifier.predict_proba(features)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> EvaluationMetrics:
        """Predict on the given features and compute standard classification metrics."""
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)[:, 1]
        return evaluate_predictions(target, predictions, probabilities)

    @property
    def feature_importances(self) -> np.ndarray:
        """Feature importances accumulated across all boosting stages."""
        return self.classifier.feature_importances_
