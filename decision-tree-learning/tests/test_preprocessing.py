"""
Tests for src/preprocessing/data_cleaner.py, feature_engineering.py, preprocessor.py.
"""

import pandas as pd

from src.data.data_loader import load_dataset
from src.preprocessing.data_cleaner import clean_dataset, fix_total_charges
from src.preprocessing.feature_engineering import create_features
from src.preprocessing.preprocessor import (
    encode_categorical_features,
    encode_target,
    split_dataset,
)


def test_fix_total_charges_produces_numeric_column():
    """After cleaning, TotalCharges should be numeric with no missing values."""
    raw_data = load_dataset()
    cleaned = fix_total_charges(raw_data)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isnull().sum() == 0


def test_fix_total_charges_fills_blanks_with_zero_for_new_customers():
    """Blank TotalCharges rows (tenure == 0) should be filled with 0.0, not the mean."""
    raw_data = load_dataset()
    cleaned = fix_total_charges(raw_data)

    new_customers = cleaned[cleaned["tenure"] == 0]
    assert (new_customers["TotalCharges"] == 0.0).all()


def test_clean_dataset_preserves_row_count():
    """This dataset has no true duplicates, so cleaning shouldn't drop rows."""
    raw_data = load_dataset()
    cleaned = clean_dataset(raw_data)

    assert len(cleaned) == len(raw_data)


def test_create_features_adds_expected_columns():
    """Feature engineering should add exactly the two documented columns."""
    raw_data = load_dataset()
    cleaned = clean_dataset(raw_data)
    engineered = create_features(cleaned)

    assert "AverageMonthlySpend" in engineered.columns
    assert "IsLongTermCustomer" in engineered.columns


def test_average_monthly_spend_has_no_infinite_values():
    """The +1 in the denominator should prevent divide-by-zero for tenure == 0 customers."""
    raw_data = load_dataset()
    cleaned = clean_dataset(raw_data)
    engineered = create_features(cleaned)

    assert not engineered["AverageMonthlySpend"].isin([float("inf"), float("-inf")]).any()


def test_is_long_term_customer_is_binary():
    """IsLongTermCustomer should only ever be 0 or 1."""
    raw_data = load_dataset()
    cleaned = clean_dataset(raw_data)
    engineered = create_features(cleaned)

    assert set(engineered["IsLongTermCustomer"].unique()).issubset({0, 1})


def test_encode_target_maps_yes_no_to_one_zero():
    """Churn should become an integer column with values 0/1 only."""
    raw_data = load_dataset()
    encoded = encode_target(raw_data)

    assert set(encoded["Churn"].unique()) == {0, 1}


def test_split_dataset_preserves_class_balance():
    """Stratified split should keep train/test churn rates close to each other."""
    raw_data = encode_target(load_dataset())
    features = raw_data.drop(columns=["customerID", "Churn"])
    target = raw_data["Churn"]

    training_features, testing_features, training_target, testing_target = split_dataset(
        features, target
    )

    assert abs(training_target.mean() - testing_target.mean()) < 0.02


def test_encode_categorical_features_produces_matching_columns():
    """Train and test encoded frames must have identical columns in identical order."""
    raw_data = encode_target(load_dataset())
    features = raw_data.drop(columns=["customerID", "Churn"])
    target = raw_data["Churn"]

    training_features, testing_features, _, _ = split_dataset(features, target)
    training_encoded, testing_encoded = encode_categorical_features(training_features, testing_features)

    assert list(training_encoded.columns) == list(testing_encoded.columns)


def test_encode_categorical_features_has_no_object_columns_left():
    """After one-hot encoding, no text/object columns should remain."""
    raw_data = encode_target(load_dataset())
    features = raw_data.drop(columns=["customerID", "Churn"])
    target = raw_data["Churn"]

    training_features, testing_features, _, _ = split_dataset(features, target)
    training_encoded, _ = encode_categorical_features(training_features, testing_features)

    assert len(training_encoded.select_dtypes(include="object").columns) == 0
