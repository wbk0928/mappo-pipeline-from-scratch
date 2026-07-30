import torch

from mappo_ss.networks.critic import Critic


STATE_DIM = 54
HIDDEN_DIM = 128

critic = Critic(
    state_dim=STATE_DIM,
    hidden_dim=HIDDEN_DIM,
)

# Batch of 4 global states
state = torch.randn(4, STATE_DIM)

values = critic(state)

print("=" * 60)
print("Input shape:")
print(state.shape)

print("=" * 60)
print("Output shape:")
print(values.shape)

print("=" * 60)
print("Values:")
print(values)