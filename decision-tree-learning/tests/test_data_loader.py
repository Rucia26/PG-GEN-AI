"""
Tests for src/data/data_loader.py and src/data/data_validator.py.

These tests use the REAL project dataset (data/raw/telco_churn.csv) rather
than a synthetic fixture, since this is a learning project — seeing tests
pass against the actual data you're working with is more concrete than
against invented rows, and the file is small enough to load quickly.
"""

import pytest

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset


def test_load_dataset_returns_nonempty_dataframe():
    """The raw dataset should load with rows and columns present."""
    dataframe = load_dataset()

    assert len(dataframe) > 0
    assert len(dataframe.columns) > 0


def test_load_dataset_has_expected_shape():
    """Regression check: catches if someone accidentally swaps the dataset file."""
    dataframe = load_dataset()

    assert dataframe.shape == (7043, 21)


def test_load_dataset_raises_for_missing_file():
    """A clear FileNotFoundError should be raised for a path that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/raw/this_file_does_not_exist.csv")


def test_expected_columns_exist():
    """Every column the rest of the pipeline assumes exists should be present."""
    dataframe = load_dataset()
    expected_columns = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"}

    assert expected_columns.issubset(set(dataframe.columns))


def test_validate_dataset_passes_on_real_data():
    """The real dataset should pass validation (warnings allowed, errors not)."""
    dataframe = load_dataset()
    result = validate_dataset(dataframe)

    assert result.is_valid is True
    assert result.errors == []


def test_validate_dataset_detects_hidden_missing_total_charges():
    """
    validate_dataset() should flag the known blank-string TotalCharges rows
    (see notebook 01) as a warning, since plain isnull() would miss them.
    """
    dataframe = load_dataset()
    result = validate_dataset(dataframe)

    assert any("TotalCharges" in warning for warning in result.warnings)


def test_validate_dataset_fails_on_missing_target_column():
    """Dropping the target column should be caught as a validation error."""
    dataframe = load_dataset().drop(columns=["Churn"])
    result = validate_dataset(dataframe)

    assert result.is_valid is False
    assert any("Churn" in error or "target" in error.lower() for error in result.errors)
