"""
Shared Actor Network for MAPPO.

The actor receives a local observation and outputs a categorical
distribution over discrete actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from mappo_ss.networks.initialization import init_layer
from mappo_ss.networks.mlp import MLP


class Actor(nn.Module):
    """
    Shared policy network.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.feature_extractor = MLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
        )

        # Small gain is standard for policy output layers.
        self.policy_head = init_layer(
            nn.Linear(hidden_dim, action_dim),
            gain=0.01,
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Build a categorical action distribution.

        Parameters
        ----------
        obs
            Shape: (..., obs_dim)

        Returns
        -------
        Categorical
            Distribution over actions.
        """

        features = self.feature_extractor(obs)

        logits = self.policy_head(features)

        return Categorical(logits=logits)