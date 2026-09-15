"""
Load and expose project-wide configuration from config.yaml.

Every other module that needs a path, a random seed, or a logging level
imports `SETTINGS` from here instead of reading config.yaml itself. This
means there is exactly one place that knows how the YAML file is structured;
if we ever rename a key in config.yaml, we only fix it in one place.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

# __file__ is this file's own path. Walking up three parents from
# src/config/settings.py lands on the project root (decision-tree-learning/).
# We compute this instead of hard-coding a path so the project works no
# matter which directory you launch Python or Jupyter from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclasses.dataclass(frozen=True)
class DataSettings:
    """Paths and column names related to the dataset."""

    raw_path: Path
    processed_path: Path
    target_column: str
    id_column: str


@dataclasses.dataclass(frozen=True)
class ModelSettings:
    """Parameters that control training/evaluation reproducibility."""

    random_state: int
    test_size: float


@dataclasses.dataclass(frozen=True)
class LoggingSettings:
    """Where and how verbosely the project logs."""

    level: str
    log_file: Path


@dataclasses.dataclass(frozen=True)
class Settings:
    """Top-level settings object grouping every config section."""

    data: DataSettings
    model: ModelSettings
    logging: LoggingSettings


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
    """
    Read config.yaml and build a validated, typed Settings object.

    Parameters
    ----------
    config_path : Path
        Location of the YAML config file. Defaults to config.yaml at the
        project root, but tests can pass a different path to load a
        temporary config without touching the real one.

    Returns
    -------
    Settings
        Immutable settings object used throughout the project.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist at config_path.
    KeyError
        If a required section or key is missing from the YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        raw_config: dict[str, Any] = yaml.safe_load(config_file)

    data_config = raw_config["data"]
    model_config = raw_config["model"]
    logging_config = raw_config["logging"]

    return Settings(
        data=DataSettings(
            raw_path=PROJECT_ROOT / data_config["raw_path"],
            processed_path=PROJECT_ROOT / data_config["processed_path"],
            target_column=data_config["target_column"],
            id_column=data_config["id_column"],
        ),
        model=ModelSettings(
            random_state=int(model_config["random_state"]),
            test_size=float(model_config["test_size"]),
        ),
        logging=LoggingSettings(
            level=logging_config["level"],
            log_file=PROJECT_ROOT / logging_config["log_file"],
        ),
    )


# Loaded once when this module is first imported, then reused everywhere:
#   from src.config.settings import SETTINGS
SETTINGS = load_settings()
