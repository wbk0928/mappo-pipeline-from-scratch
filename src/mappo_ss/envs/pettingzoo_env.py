"""
Generic PettingZoo environment wrapper.

Currently supports:
    - simple_spread_v3 (MPE2)

Designed for CTDE (Centralized Training, Decentralized Execution).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Discrete
from mpe2 import simple_spread_v3

from mappo_ss.config.config import Config


class PettingZooEnv:
    """Generic wrapper for PettingZoo parallel environments."""

    def __init__(self, cfg: Config):

        env_cfg = cfg["environment"]

        env_name = env_cfg["env_name"]

        if env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(
                N=env_cfg["num_agents"],
                max_cycles=env_cfg["max_cycles"],
                continuous_actions=env_cfg["continuous_actions"],
                render_mode=env_cfg["render_mode"],
            )
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        self.agent_ids: list[str] = []
        self.num_agents: int = 0

        self.obs_dim: int = 0
        self.action_dim: int = 0
        self.state_dim: int = 0

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:

        observations, infos = self.env.reset(seed=seed)

        self.agent_ids = list(self.env.agents)
        self.num_agents = len(self.agent_ids)

        first_agent = self.agent_ids[0]

        self.obs_dim = observations[first_agent].shape[0]

        action_space = self.env.action_space(first_agent)

        if not isinstance(action_space, Discrete):
            raise RuntimeError(
                "Current MAPPO implementation supports only discrete actions."
            )

        self.action_dim = action_space.n

        self.state_dim = self.obs_dim * self.num_agents

        return observations, infos

    def step(self, actions):

        return self.env.step(actions)

    def get_global_state(
        self,
        observations: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Build centralized state by concatenating
        every agent observation.
        """

        return np.concatenate(
            [observations[agent] for agent in self.agent_ids],
            axis=0,
        )

    def close(self):

        self.env.close()