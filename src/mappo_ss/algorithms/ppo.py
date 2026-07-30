"""
PPO update step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mappo_ss.algorithms.losses import (
    clipped_policy_loss,
    entropy_bonus,
    value_loss,
)


class PPOUpdater:
    """
    Performs one PPO optimization step.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        value_normalizer,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
    ):

        self.actor = actor
        self.critic = critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_normalizer= value_normalizer

        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        observations,
        global_states,
        actions,
        old_log_probs,
        old_values,
        advantages,
        returns,
    ):
        """
        Perform one PPO update.
        """

        dist = self.actor(observations)

        new_log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        values = self.critic(global_states)

        # Update running statistics

        normalized_values = self.value_normalizer.normalize(values)

        normalized_old_values = old_values

        normalized_returns = self.value_normalizer.normalize(
            returns
        )
        


        policy_loss = clipped_policy_loss(
            old_log_probs,
            new_log_probs,
            advantages,
            self.clip_ratio,
        )



        critic_loss = value_loss(
            predicted_values=normalized_values,
            old_values=normalized_old_values,
            returns=normalized_returns,
            clip_ratio=self.clip_ratio,
        )

        entropy_loss = entropy_bonus(entropy)

        actor_loss = (
            policy_loss
            - self.entropy_coef * entropy_loss
        )

        total_critic_loss = (
            self.value_coef * critic_loss
        )

        # ---------------------
        # Actor update
        # ---------------------

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.max_grad_norm,
        )

        self.actor_optimizer.step()

        # ---------------------
        # Critic update
        # ---------------------

        self.critic_optimizer.zero_grad()

        total_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
        )

        self.critic_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item(),
        }