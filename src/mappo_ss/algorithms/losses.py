"""
PPO loss functions.
"""

from __future__ import annotations

import torch


def clipped_policy_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    """
    PPO clipped surrogate objective.
    """

    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages

    clipped = (
        torch.clamp(
            ratio,
            1.0 - clip_ratio,
            1.0 + clip_ratio,
        )
        * advantages
    )

    loss = -torch.min(unclipped, clipped).mean()

    return loss


def value_loss(
    predicted_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    """
    PPO clipped value loss.
    """

    value_clipped = old_values + (
        predicted_values - old_values
    ).clamp(
        -clip_ratio,
        clip_ratio,
    )

    value_loss_unclipped = (
        predicted_values - returns
    ).pow(2)

    value_loss_clipped = (
        value_clipped - returns
    ).pow(2)

    loss = torch.max(
        value_loss_unclipped,
        value_loss_clipped,
    )

    return loss.mean()

def entropy_bonus(
    entropy: torch.Tensor,
) -> torch.Tensor:
    """
    Entropy bonus.
    """

    return entropy.mean()