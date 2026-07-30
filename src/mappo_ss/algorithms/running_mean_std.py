"""
Running mean and variance for observation normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):

    def __init__(self, shape, epsilon=1e-4, device="cpu"):

        super().__init__()

        self.register_buffer(
            "mean",
            torch.zeros(shape, device=device),
        )

        self.register_buffer(
            "var",
            torch.ones(shape, device=device),
        )

        self.register_buffer(
            "count",
            torch.tensor(epsilon, device=device),
        )

    @torch.no_grad()
    def update(self, x):

        x = x.view(-1, x.shape[-1])

        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        M2 = (
            m_a
            + m_b
            + delta.pow(2)
            * self.count
            * batch_count
            / total_count
        )

        new_var = M2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x):

        return (
            x - self.mean
        ) / torch.sqrt(
            self.var + 1e-8
        )