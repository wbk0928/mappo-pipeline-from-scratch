"""
Generic multi-layer perceptron.

Used by both the Actor and Critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mappo_ss.networks.initialization import init_layer


class MLP(nn.Module):
    """
    Two-layer MLP feature extractor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_dim), gain=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim), gain=2**0.5),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Feature representation.
        """

        return self.network(x)