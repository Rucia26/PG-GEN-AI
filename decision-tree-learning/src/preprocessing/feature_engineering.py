"""
Create derived features for the churn dataset.

Feature engineering means building new columns from existing ones that make
a pattern easier for the model to find than the raw columns alone. Every
function here is deliberately simple and explainable — the point, for a
learning project, is to see the *reasoning* behind each feature, not to
maximize leaderboard score with opaque transformations.

IMPORTANT — leakage warning:
Every feature built here uses only information available at the time of
prediction (existing customer attributes). None of them use the target
(Churn) or anything computed using future information. That's what makes
them safe to compute before the train/test split, unlike encoders/scalers
which must be fit on the training set only (see preprocessor.py).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_average_monthly_spend(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add AverageMonthlySpend = TotalCharges / (tenure + 1).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset containing numeric 'TotalCharges' and 'tenure'.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an added 'AverageMonthlySpend' column.

    Notes
    -----
    We add 1 to tenure to prevent division-by-zero for brand-new customers
    (tenure == 0), rather than filtering them out or special-casing them.
    This is a small but real example of a "what could go wrong" case: a
    naive `TotalCharges / tenure` would raise a ZeroDivisionError-equivalent
    (produce inf/NaN in pandas) for every new customer.
    """
    engineered = dataframe.copy()
    engineered["AverageMonthlySpend"] = engineered["TotalCharges"] / (engineered["tenure"] + 1)
    return engineered


def add_is_long_term_customer(dataframe: pd.DataFrame, tenure_threshold_months: int = 24) -> pd.DataFrame:
    """
    Add a binary flag IsLongTermCustomer = tenure >= tenure_threshold_months.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset containing 'tenure'.
    tenure_threshold_months : int
        Minimum tenure (in months) to be considered "long-term". Default of
        24 months (2 years) matches the longest contract type in this
        dataset, so the flag roughly asks "has this customer already
        outlasted a full long-term contract?".

    Returns
    -------
    pd.DataFrame
        Copy of the input with an added 'IsLongTermCustomer' column (0/1).

    Notes
    -----
    A decision tree CAN discover an equivalent split on 'tenure' by itself
    (e.g. "tenure >= 24") — so this feature doesn't add information a tree
    couldn't already find. We include it anyway because it demonstrates the
    concept of feature engineering clearly and gives models that DON'T
    automatically find thresholds (e.g. linear models, if you experiment
    with them later) direct access to the same signal.
    """
    engineered = dataframe.copy()
    engineered["IsLongTermCustomer"] = (engineered["tenure"] >= tenure_threshold_months).astype(int)
    return engineered


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset (post data_cleaner.clean_dataset()).

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added.
    """
    engineered = add_average_monthly_spend(dataframe)
    engineered = add_is_long_term_customer(engineered)
    logger.info(
        "Feature engineering complete. Added columns: %s",
        ["AverageMonthlySpend", "IsLongTermCustomer"],
    )
    return engineered
