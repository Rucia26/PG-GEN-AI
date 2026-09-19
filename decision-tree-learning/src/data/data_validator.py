"""
Validate the structure and basic quality of the raw dataset.

This module answers "is this dataset shaped the way the rest of the
pipeline expects?" before any cleaning or modeling happens. Catching a
problem here, with a clear error message, is far cheaper than discovering
it three steps later as a cryptic KeyError inside model training.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# The Telco churn dataset encodes its target as these two literal strings.
# If this ever needs to change, it changes here and nowhere else.
_VALID_TARGET_VALUES = {"Yes", "No"}


def validate_dataset(
    dataframe: pd.DataFrame,
    expected_columns: Iterable[str],
    target_column: str,
) -> None:
    """
    Check that the dataset has the columns and target values we expect.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The raw, just-loaded dataset (before cleaning).
    expected_columns : Iterable[str]
        Column names that must be present in the dataset.
    target_column : str
        Name of the column containing the churn label.

    Returns
    -------
    None
        This function does not transform the data — it only checks it and
        raises/logs. Cleaning happens in src/preprocessing/.

    Raises
    ------
    KeyError
        If any expected column, including the target column, is missing.
    ValueError
        If the target column contains values other than "Yes"/"No".
    """
    missing_columns = set(expected_columns) - set(dataframe.columns)
    if missing_columns:
        logger.error("Dataset is missing expected columns: %s", missing_columns)
        raise KeyError(f"Missing expected columns: {missing_columns}")

    if target_column not in dataframe.columns:
        logger.error("Target column '%s' not found in dataset", target_column)
        raise KeyError(f"Target column '{target_column}' not found in dataset")

    unexpected_target_values = set(dataframe[target_column].unique()) - _VALID_TARGET_VALUES
    if unexpected_target_values:
        logger.error(
            "Target column '%s' contains unexpected values: %s",
            target_column,
            unexpected_target_values,
        )
        raise ValueError(
            f"Target column '{target_column}' should only contain "
            f"{_VALID_TARGET_VALUES}, found: {unexpected_target_values}"
        )

    _warn_about_missing_values(dataframe)
    _warn_about_duplicate_rows(dataframe)
    _warn_about_hidden_numeric_missing_values(dataframe)

    logger.info("Dataset validation passed: %d rows, %d columns", *dataframe.shape)


def _warn_about_missing_values(dataframe: pd.DataFrame) -> None:
    """Log a warning for any column containing true NaN values."""
    missing_value_counts = dataframe.isnull().sum()
    columns_with_missing_values = missing_value_counts[missing_value_counts > 0]
    if not columns_with_missing_values.empty:
        logger.warning("Missing (NaN) values detected:\n%s", columns_with_missing_values.to_string())


def _warn_about_duplicate_rows(dataframe: pd.DataFrame) -> None:
    """Log a warning if any fully duplicated rows exist."""
    duplicate_row_count = int(dataframe.duplicated().sum())
    if duplicate_row_count > 0:
        logger.warning("Found %d fully duplicated rows", duplicate_row_count)


def _warn_about_hidden_numeric_missing_values(dataframe: pd.DataFrame) -> None:
    """
    Detect missing values disguised as blank/whitespace strings.

    pandas' isnull() only catches true NaN. A column like the Telco
    dataset's TotalCharges can be stored as text (e.g. because a handful
    of rows contain a single space " " instead of a number) and isnull()
    will report zero missing values even though those rows are unusable
    as numbers. We check this by looking at text columns that are "mostly
    numeric" and seeing whether converting them to numbers produces MORE
    missing values than isnull() already found.
    """
    text_columns = dataframe.select_dtypes(include=["object", "string"]).columns
    for column_name in text_columns:
        coerced_to_numeric = pd.to_numeric(dataframe[column_name], errors="coerce")
        mostly_numeric = coerced_to_numeric.notna().mean() > 0.9
        if not mostly_numeric:
            continue

        true_missing_count = dataframe[column_name].isnull().sum()
        hidden_missing_count = int(coerced_to_numeric.isna().sum() - true_missing_count)
        if hidden_missing_count > 0:
            logger.warning(
                "Column '%s' looks numeric but is stored as text; %d value(s) are "
                "blank/non-numeric rather than true NaN, so isnull() misses them. "
                "This needs explicit handling during preprocessing.",
                column_name,
                hidden_missing_count,
            )
