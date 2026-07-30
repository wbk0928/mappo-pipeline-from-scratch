"""
Logging utilities.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger