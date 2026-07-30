import torch

from mappo_ss.networks.actor import Actor


OBS_DIM = 18
HIDDEN_DIM = 128
ACTION_DIM = 5

actor = Actor(
    obs_dim=OBS_DIM,
    hidden_dim=HIDDEN_DIM,
    action_dim=ACTION_DIM,
)

# Batch of 4 observations
obs = torch.randn(4, OBS_DIM)

dist = actor(obs)

actions = dist.sample()
log_probs = dist.log_prob(actions)
entropy = dist.entropy()

print("=" * 60)
print("Actions:")
print(actions)

print("=" * 60)
print("Log probabilities:")
print(log_probs)

print("=" * 60)
print("Entropy:")
print(entropy)