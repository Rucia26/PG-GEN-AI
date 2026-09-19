"""
Encode categorical features and split the dataset into train/test sets.

WHY THIS MODULE FITS ENCODERS ON THE TRAINING SET ONLY
--------------------------------------------------------
A core rule in ML: never let information from the test set influence
training. If you one-hot encode or scale using statistics computed from the
WHOLE dataset (train + test combined), the test set has quietly "leaked"
information into the model before you ever evaluate it — your test accuracy
will look better than the model's true performance on genuinely unseen data.

The functions below always follow the same order:
    1. split_dataset()          — split FIRST, before fitting anything
    2. encode_categorical_features() — fit encoding on TRAINING data only,
                                        then apply that same mapping to test
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import SETTINGS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def encode_target(dataframe: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Encode the target column from 'Yes'/'No' strings to 1/0 integers.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset containing the target column as 'Yes'/'No' strings.
    target_column : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with target_column encoded as int (1 = Yes/churn,
        0 = No/stayed).

    Notes
    -----
    The target is encoded on the FULL dataset (before splitting) because
    "Yes" always means 1 and "No" always means 0 — this mapping doesn't
    depend on any statistic of the data, so there's no leakage risk here,
    unlike the feature encoding below which does depend on the data.
    """
    encoded = dataframe.copy()
    encoded[target_column] = encoded[target_column].map({"Yes": 1, "No": 0}).astype(int)
    return encoded


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.

    Parameters
    ----------
    features : pd.DataFrame
        All predictor columns (no target, no ID column).
    target : pd.Series
        The encoded target column (0/1).

    Returns
    -------
    tuple
        (training_features, testing_features, training_target, testing_target)

    Notes
    -----
    stratify=target keeps the same churn ratio (~26.5% / 73.5%, from
    notebook 01) in both the training and test sets. Without stratification,
    a random split could — by chance — put an unusually high or low
    proportion of churners in the test set, making evaluation results noisy
    and harder to compare across experiments.
    """
    training_features, testing_features, training_target, testing_target = train_test_split(
        features,
        target,
        test_size=SETTINGS.model.test_size,
        random_state=SETTINGS.model.random_state,
        stratify=target,
    )

    logger.info(
        "Split dataset: %d training rows, %d testing rows (test_size=%s)",
        len(training_features),
        len(testing_features),
        SETTINGS.model.test_size,
    )

    return training_features, testing_features, training_target, testing_target


def encode_categorical_features(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical columns, fitting the category set on training data only.

    Parameters
    ----------
    training_features : pd.DataFrame
        Training split predictors (output of split_dataset).
    testing_features : pd.DataFrame
        Testing split predictors (output of split_dataset).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (encoded_training_features, encoded_testing_features), with matching
        columns in matching order.

    Notes
    -----
    pandas.get_dummies() run separately on train and test could produce
    DIFFERENT columns if a category appears in one split but not the other
    (e.g. a rare PaymentMethod value only present in training). We guard
    against this by reindexing the test set onto the training set's exact
    column list, filling any missing dummy columns with 0. This is the
    encoding equivalent of "fit on train, transform on test."
    """
    categorical_columns = training_features.select_dtypes(include="object").columns.tolist()
    logger.info("One-hot encoding %d categorical column(s)", len(categorical_columns))

    encoded_training = pd.get_dummies(training_features, columns=categorical_columns)
    encoded_testing = pd.get_dummies(testing_features, columns=categorical_columns)

    # Align test columns to the training set's columns (fit-on-train
    # discipline). Any category seen only in training gets a 0 column;
    # any category seen only in test is silently dropped, since the model
    # was never trained with a corresponding column for it.
    encoded_testing = encoded_testing.reindex(columns=encoded_training.columns, fill_value=0)

    return encoded_training, encoded_testing
