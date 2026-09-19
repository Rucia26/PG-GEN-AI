"""
Load the raw customer churn dataset from disk.

This module is intentionally small: its only job is turning a CSV file into
a pandas DataFrame and telling us what happened. Cleaning, encoding, and
feature engineering all live in src/preprocessing/ — keeping "read the file"
separate from "fix the file" makes each step independently testable and
easy to reason about.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load the customer churn dataset from a CSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to the input CSV file (e.g. SETTINGS.data.raw_path).

    Returns
    -------
    pd.DataFrame
        The dataset exactly as stored on disk — no cleaning applied yet.

    Raises
    ------
    FileNotFoundError
        If no file exists at file_path. This is deliberately allowed to
        propagate (rather than being caught and hidden) so that a missing
        dataset fails loudly and immediately, instead of causing a
        confusing error several steps later in the pipeline.
    """
    file_path = Path(file_path)
    logger.info("Loading dataset from %s", file_path)

    if not file_path.exists():
        logger.error("Dataset file not found at %s", file_path)
        raise FileNotFoundError(
            f"No dataset found at {file_path}. "
            "Place the Telco Customer Churn CSV there before running the pipeline."
        )

    dataframe = pd.read_csv(file_path)
    logger.info("Dataset loaded successfully. Shape: %s", dataframe.shape)

    return dataframe
