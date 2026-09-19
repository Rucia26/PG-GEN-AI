"""
Load the raw Telco Customer Churn dataset from disk.

This module has exactly one responsibility: turn a CSV file on disk into a
pandas DataFrame. It does not clean data, engineer features, or validate
business rules — that separation means a bug in "is this CSV readable" can
never be confused with a bug in "does this data make sense" (data_validator.py
owns that instead).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import SETTINGS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the customer churn dataset from a CSV file.

    Parameters
    ----------
    file_path : str | Path | None
        Path to the input CSV file. Defaults to the raw data path configured
        in config.yaml (SETTINGS.data.raw_path) so callers don't need to know
        the path themselves.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified customer dataset — exactly what is on disk.

    Raises
    ------
    FileNotFoundError
        If no file exists at the resolved path. Raised explicitly (rather
        than letting pandas' own FileNotFoundError propagate silently) so the
        error message tells you which path was actually checked, since the
        path is often built from config rather than typed directly.
    """
    resolved_path = Path(file_path) if file_path is not None else SETTINGS.data.raw_path

    if not resolved_path.exists():
        logger.error("Dataset not found at %s", resolved_path)
        raise FileNotFoundError(
            f"No dataset file at '{resolved_path}'. "
            "Did you download telco_churn.csv into data/raw/?"
        )

    logger.info("Loading dataset from %s", resolved_path)
    dataframe = pd.read_csv(resolved_path)
    logger.info("Dataset loaded successfully. Shape: %s", dataframe.shape)

    return dataframe
