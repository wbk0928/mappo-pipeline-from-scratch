"""
Data structure representing one environment timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transition:
    """
    One timestep collected from the environment.
    """

    observations: dict[str, np.ndarray]

    global_state: np.ndarray

    actions: dict[str, int]

    log_probs: dict[str, float]

    rewards: dict[str, float]

    values: dict[str, float]

    next_observations: dict[str, np.ndarray]

    next_global_state: np.ndarray

    terminations: dict[str, bool]

    truncations: dict[str, bool]