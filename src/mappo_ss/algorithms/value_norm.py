"""
Running value normalization used by MAPPO.

Normalizes critic targets using running statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """
    Running mean and variance for value targets.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.zeros(1),
        )

        self.register_buffer(
            "var",
            torch.ones(1),
        )

        self.register_buffer(
            "count",
            torch.tensor(epsilon),
        )

    @torch.no_grad()
    def update(
        self,
        values: torch.Tensor,
    ):
        """
        Update running statistics.
        """

        values = values.view(-1)

        batch_mean = values.mean()

        batch_var = values.var(unbiased=False)

        batch_count = values.numel()

        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        m_a = self.var * self.count

        m_b = batch_var * batch_count

        m2 = (
            m_a
            + m_b
            + delta.pow(2)
            * self.count
            * batch_count
            / total_count
        )

        new_var = m2 / total_count

        self.mean.copy_(new_mean)

        self.var.copy_(new_var)

        self.count.copy_(total_count)

    def normalize(
        self,
        values: torch.Tensor,
    ):
        """
        Normalize values.
        """

        return (
            values - self.mean
        ) / torch.sqrt(
            self.var + 1e-8
        )

    def denormalize(
        self,
        values: torch.Tensor,
    ):
        """
        Recover original scale.
        """

        return (
            values
            * torch.sqrt(self.var + 1e-8)
            + self.mean
        )