"""
Centralized Critic for MAPPO.

The critic receives the global state and predicts
the state value V(s).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mappo_ss.networks.initialization import init_layer
from mappo_ss.networks.mlp import MLP


class Critic(nn.Module):
    """
    Centralized value network.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.feature_extractor = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        # Standard PPO/MAPPO initialization for value head
        self.value_head = init_layer(
            nn.Linear(hidden_dim, 1),
            gain=1.0,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict V(s).

        Parameters
        ----------
        state
            Shape: (..., state_dim)

        Returns
        -------
        torch.Tensor
            Shape: (...,)
        """

        features = self.feature_extractor(state)

        value = self.value_head(features)

        return value.squeeze(-1)