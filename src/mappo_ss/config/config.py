"""
Configuration loader.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Simple wrapper around a configuration dictionary."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def __getitem__(self, key: str):
        return self._config[key]

    def as_dict(self):
        return self._config


def load_config(path: str | Path) -> Config:
    """Load YAML configuration."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return Config(config)