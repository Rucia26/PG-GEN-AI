"""
Validate structural assumptions about the raw churn dataset.

Why a separate module from data_loader.py?
-------------------------------------------
Loading answers "can I read this file?". Validation answers "is this the
data I expect?". Keeping them apart means a schema change (a renamed column,
a new category) fails loudly and specifically here, instead of surfacing
later as a confusing KeyError deep inside feature engineering or model
training.

This module only *reports* problems via the returned ValidationResult and
log messages — it never mutates the DataFrame or fixes anything. Fixing
belongs in src/preprocessing/data_cleaner.py.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from src.config.settings import SETTINGS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The columns we expect the raw CSV to contain. Defined here (not inferred
# from the DataFrame) so that a silently dropped or renamed column is
# detected immediately, rather than causing a KeyError three notebooks later.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
)

EXPECTED_TARGET_VALUES: frozenset[str] = frozenset({"Yes", "No"})


@dataclasses.dataclass
class ValidationResult:
    """
    Outcome of validating a DataFrame against our schema expectations.

    is_valid is False only for problems that would break downstream code
    (missing columns, missing target). Data-quality quirks that are expected
    and handled later (like blank TotalCharges strings) are reported as
    warnings, not failures — see the distinction explained in
    validate_dataset() below.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_dataset(dataframe: pd.DataFrame) -> ValidationResult:
    """
    Check that a loaded DataFrame matches the schema this project expects.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The raw dataset, as returned by data_loader.load_dataset().

    Returns
    -------
    ValidationResult
        is_valid=False means the DataFrame is unsafe to continue with
        (e.g. missing the target column). Warnings describe known, handleable
        data-quality issues (e.g. blank TotalCharges) that preprocessing will
        fix later — they don't block the pipeline, but you should read them.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = set(EXPECTED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        errors.append(f"Missing expected columns: {sorted(missing_columns)}")

    target_column = SETTINGS.data.target_column
    if target_column not in dataframe.columns:
        errors.append(f"Target column '{target_column}' not found in dataset")
    else:
        unexpected_targets = set(dataframe[target_column].unique()) - EXPECTED_TARGET_VALUES
        if unexpected_targets:
            errors.append(
                f"Target column contains unexpected values: {sorted(unexpected_targets)}"
            )

    id_column = SETTINGS.data.id_column
    if id_column in dataframe.columns:
        duplicate_count = dataframe[id_column].duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"{duplicate_count} duplicate '{id_column}' values found")

    # TotalCharges is loaded as a string column because a small number of
    # brand-new customers (tenure == 0) have a BLANK STRING instead of a
    # number. pandas' isnull() does not catch this — "" is not NaN — so a
    # naive `dataframe['TotalCharges'].isnull().sum()` reports zero missing
    # values even though the column is not truly numeric yet. We check for
    # this explicitly rather than let it surface later as a confusing
    # "could not convert string to float" error during model training.
    if "TotalCharges" in dataframe.columns:
        non_numeric = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
        blank_count = non_numeric.isnull().sum()
        if blank_count > 0:
            warnings.append(
                f"{blank_count} rows have a non-numeric TotalCharges "
                "(typically blank strings for tenure == 0 customers). "
                "This is expected - data_cleaner.py handles it."
            )

    for warning in warnings:
        logger.warning(warning)
    for error in errors:
        logger.error(error)

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Dataset validation passed (%d warning(s))", len(warnings))

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
