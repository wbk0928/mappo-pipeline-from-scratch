import torch

from mappo_ss.algorithms.losses import (
    clipped_policy_loss,
    value_loss,
    entropy_bonus,
)

old_log_probs = torch.tensor([-1.2, -0.8, -0.5])
new_log_probs = torch.tensor([-1.1, -0.7, -0.6])

advantages = torch.tensor([1.0, 0.5, -0.3])

policy = clipped_policy_loss(
    old_log_probs,
    new_log_probs,
    advantages,
    clip_ratio=0.2,
)

values = torch.tensor([1.2, 0.9, 0.3])
returns = torch.tensor([1.0, 1.0, 0.0])

critic = value_loss(
    values,
    returns,
)

entropy = entropy_bonus(
    torch.tensor([1.60, 1.58, 1.61]),
)

print("=" * 60)
print("Policy loss:", policy.item())

print("=" * 60)
print("Value loss:", critic.item())

print("=" * 60)
print("Entropy:", entropy.item())