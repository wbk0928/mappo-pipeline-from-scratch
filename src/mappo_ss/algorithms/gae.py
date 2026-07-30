"""
Generalized Advantage Estimation (GAE).
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : (T,)
    values : (T,)
    dones : (T,)
    next_value : scalar
    """

    advantages = torch.zeros_like(rewards)

    gae = 0.0

    for t in reversed(range(len(rewards))):

        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        mask = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_val * mask - values[t]

        gae = delta + gamma * gae_lambda * mask * gae

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns