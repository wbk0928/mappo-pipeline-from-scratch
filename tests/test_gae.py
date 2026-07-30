import torch

from mappo_ss.algorithms.gae import compute_gae

rewards = torch.tensor([1., 1., 1., 1.])
values = torch.tensor([0.5, 0.6, 0.7, 0.8])
dones = torch.tensor([0., 0., 0., 1.])

next_value = torch.tensor(0.)

advantages, returns = compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    next_value=next_value,
    gamma=0.99,
    gae_lambda=0.95,
)

print("=" * 60)
print("Advantages:")
print(advantages)

print("=" * 60)
print("Returns:")
print(returns)