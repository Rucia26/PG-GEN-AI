"""
Tests for src/models/*.py.

We test the wrapper classes on a small synthetic dataset rather than the
full churn dataset — fast to run, and it keeps the tests focused on
"does fit/predict/predict_proba work correctly," not on churn-specific
performance thresholds (that's what the notebooks explore).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.decision_tree_model import DecisionTreeModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel


@pytest.fixture
def toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A tiny, clearly-separable binary classification dataset."""
    rng = np.random.RandomState(42)
    feature_one = rng.normal(loc=0, scale=1, size=200)
    feature_two = rng.normal(loc=0, scale=1, size=200)
    target = (feature_one + feature_two > 0).astype(int)

    features = pd.DataFrame({"feature_one": feature_one, "feature_two": feature_two})
    return features, pd.Series(target, name="target")


@pytest.mark.parametrize("model_class", [DecisionTreeModel, RandomForestModel, GradientBoostingModel])
def test_model_can_train(model_class, toy_dataset):
    """Every model wrapper should fit without raising."""
    features, target = toy_dataset
    model = model_class()
    model.fit(features, target)  # should not raise


@pytest.mark.parametrize("model_class", [DecisionTreeModel, RandomForestModel, GradientBoostingModel])
def test_model_can_predict(model_class, toy_dataset):
    """predict() should return one label per input row."""
    features, target = toy_dataset
    model = model_class()
    model.fit(features, target)

    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert set(np.unique(predictions)).issubset({0, 1})


@pytest.mark.parametrize("model_class", [DecisionTreeModel, RandomForestModel, GradientBoostingModel])
def test_predict_proba_shape_is_correct(model_class, toy_dataset):
    """predict_proba() should return (n_samples, 2) probabilities that sum to 1."""
    features, target = toy_dataset
    model = model_class()
    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(features), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("model_class", [DecisionTreeModel, RandomForestModel, GradientBoostingModel])
def test_evaluate_returns_metrics_in_valid_range(model_class, toy_dataset):
    """All returned metrics should be valid probabilities/scores in [0, 1]."""
    features, target = toy_dataset
    model = model_class()
    model.fit(features, target)

    metrics = model.evaluate(features, target)

    for metric_name, metric_value in metrics.to_dict().items():
        assert 0.0 <= metric_value <= 1.0, f"{metric_name} out of range: {metric_value}"


def test_decision_tree_learns_the_separable_pattern(toy_dataset):
    """
    On a clearly linearly-separable toy problem, a decision tree should
    achieve near-perfect training accuracy — a sanity check that fit()
    actually learns something, not just that it runs without error.
    """
    features, target = toy_dataset
    model = DecisionTreeModel(max_depth=5)
    model.fit(features, target)

    metrics = model.evaluate(features, target)

    assert metrics.accuracy > 0.9
