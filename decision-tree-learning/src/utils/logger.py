"""
Centralized logging setup, shared by every module in the project.

Why not just use print()?
--------------------------
print() has no timestamp, no severity level, no record of which module
produced the message, and no way to route messages to a file for later
debugging. logging gives us all four, and lets us silence DEBUG-level noise
in production while keeping it available during development, just by
changing one config value (logging.level in config.yaml).
"""

from __future__ import annotations

import logging

from src.config.settings import SETTINGS

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(module_name: str) -> logging.Logger:
    """
    Create (or retrieve) a logger configured for console + file output.

    Parameters
    ----------
    module_name : str
        Name to tag log messages with, conventionally `__name__` of the
        calling module (e.g. "src.data.data_loader").

    Returns
    -------
    logging.Logger
        A logger that writes to both the console and logs/ml_project.log,
        using the level configured in config.yaml.
    """
    logger = logging.getLogger(module_name)

    # logging.getLogger(name) returns the SAME object every time it's called
    # with the same name. Without this guard, re-importing a module (common
    # in Jupyter when you re-run a cell) would attach a second console
    # handler and a second file handler, causing every log line to print
    # twice, then three times, and so on.
    if logger.handlers:
        return logger

    logger.setLevel(SETTINGS.logging.level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file_path = SETTINGS.logging.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent messages from also being handled by the root logger, which
    # would otherwise duplicate output if the root logger is ever configured.
    logger.propagate = False

    return logger
