"""
Fix known data-quality issues in the raw churn dataset.

This module ONLY fixes problems that data_validator.py already knows how to
detect (see EXPECTED_COLUMNS / the TotalCharges check there). It does not
invent new features (that's feature_engineering.py) and does not encode
categories into numbers (that's preprocessor.py) — each module has exactly
one job, which makes it possible to test and reason about them independently.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fix_total_charges(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges from text to numeric, filling blank entries.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset containing a 'TotalCharges' column, as loaded by
        data_loader.load_dataset().

    Returns
    -------
    pd.DataFrame
        A copy of the input with 'TotalCharges' as float64 and no blanks.

    Notes
    -----
    As found during data understanding (notebook 01), every row with a blank
    TotalCharges has tenure == 0 — a customer who just signed up and hasn't
    been billed yet. The mathematically honest value for "total charged to a
    customer with zero tenure" is 0.0, not the column mean or median (which
    would fabricate billing history that never happened).
    """
    cleaned = dataframe.copy()

    charges_numeric = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    missing_count = charges_numeric.isnull().sum()

    if missing_count > 0:
        logger.info(
            "Converting TotalCharges to numeric; filling %d blank value(s) with 0.0 "
            "(all correspond to tenure == 0 customers)",
            missing_count,
        )
    cleaned["TotalCharges"] = charges_numeric.fillna(0.0)

    return cleaned


def remove_duplicate_customers(dataframe: pd.DataFrame, id_column: str = "customerID") -> pd.DataFrame:
    """
    Drop rows with a duplicate customer ID, keeping the first occurrence.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset containing the id_column.
    id_column : str
        Name of the unique customer identifier column.

    Returns
    -------
    pd.DataFrame
        A copy of the input with duplicate IDs removed. If there are no
        duplicates (as is the case for this dataset — see notebook 01), the
        returned frame is unchanged in content.
    """
    duplicate_count = dataframe[id_column].duplicated().sum()
    if duplicate_count > 0:
        logger.warning("Removing %d duplicate '%s' row(s)", duplicate_count, id_column)
    else:
        logger.info("No duplicate '%s' values found", id_column)

    return dataframe.drop_duplicates(subset=id_column, keep="first").copy()


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline: fix TotalCharges, then drop duplicates.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw dataset, as loaded by data_loader.load_dataset().

    Returns
    -------
    pd.DataFrame
        Cleaned dataset, safe to pass into feature engineering / encoding.
    """
    cleaned = fix_total_charges(dataframe)
    cleaned = remove_duplicate_customers(cleaned)
    logger.info("Cleaning complete. Final shape: %s", cleaned.shape)
    return cleaned
