"""
Weight initialization utilities.

MAPPO (and PPO in general) commonly use orthogonal initialization
for improved training stability.
"""

from __future__ import annotations

import torch.nn as nn


def init_layer(
    layer: nn.Module,
    gain: float = 1.0,
    bias_const: float = 0.0,
) -> nn.Module:
    """
    Orthogonally initialize a layer.

    Parameters
    ----------
    layer
        Linear layer.
    gain
        Gain used for orthogonal initialization.
    bias_const
        Constant used to initialize the bias.

    Returns
    -------
    nn.Module
        Initialized layer.
    """

    nn.init.orthogonal_(layer.weight, gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)

    return layer