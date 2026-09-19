"""
Reusable plotting functions for model results.

Notebooks 04-07 built these plots inline first, so you could see the exact
matplotlib/seaborn calls next to the concept being taught. This module
extracts the versions worth reusing across notebooks (and, later, scripts)
so a plot's styling only needs to be fixed in one place.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(confusion: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
    """
    Draw a confusion matrix heatmap for a binary classifier.

    Parameters
    ----------
    confusion : np.ndarray
        2x2 confusion matrix, as returned by
        sklearn.metrics.confusion_matrix or EvaluationMetrics.confusion_matrix.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The created figure, so callers can further customize or save it.
    """
    figure, axes = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion, annot=True, fmt="d", cmap="Blues", ax=axes,
        xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(title)
    return figure


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 10,
    title: str = "Feature Importance",
) -> plt.Figure:
    """
    Draw a horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    feature_names : list[str]
        Column names corresponding to `importances`.
    importances : np.ndarray
        Feature importance scores (e.g. model.feature_importances_).
    top_n : int
        Number of top features to display.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    axes.set_xlabel("Importance")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def plot_model_complexity(complexity_df: pd.DataFrame) -> plt.Figure:
    """
    Plot training vs. testing accuracy across model complexity values.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Must contain 'max_depth', 'training_accuracy', 'testing_accuracy'
        columns, as produced in notebook 04's overfitting experiment.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    figure, axes = plt.subplots(figsize=(8, 5))
    x_positions = range(len(complexity_df))
    axes.plot(x_positions, complexity_df["training_accuracy"], marker="o", label="Training Accuracy")
    axes.plot(x_positions, complexity_df["testing_accuracy"], marker="o", label="Testing Accuracy")
    axes.set_xticks(list(x_positions))
    axes.set_xticklabels(complexity_df["max_depth"])
    axes.set_xlabel("max_depth")
    axes.set_ylabel("Accuracy")
    axes.set_title("Model Complexity vs. Performance")
    axes.legend()
    axes.grid(True, alpha=0.3)
    return figure


def plot_decision_tree(classifier, feature_names: list[str], max_depth: int | None = 2) -> plt.Figure:
    """
    Visualize a fitted DecisionTreeClassifier.

    Parameters
    ----------
    classifier : sklearn.tree.DecisionTreeClassifier
        A FITTED tree (call .fit() first).
    feature_names : list[str]
        Column names used to train the tree.
    max_depth : int | None
        How many levels deep to render. Full trees are often too dense to
        read — defaults to 2 levels, matching notebook 04's visualization.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    figure, axes = plt.subplots(figsize=(20, 10))
    plot_tree(
        classifier,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=axes,
    )
    return figure


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Plot a grouped bar chart comparing metrics across multiple models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Rows = model names, columns = metric names (accuracy, precision,
        recall, f1, roc_auc) — e.g. notebook 07's final_comparison table.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    axes = comparison_df.plot(kind="bar", figsize=(11, 5), rot=0)
    axes.set_title("Model Comparison — All Metrics")
    axes.set_ylabel("Score")
    axes.legend(loc="lower right")
    figure = axes.get_figure()
    figure.tight_layout()
    return figure
