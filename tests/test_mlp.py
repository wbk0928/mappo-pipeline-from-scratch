import torch

from mappo_ss.networks.mlp import MLP


model = MLP(
    input_dim=18,
    hidden_dim=128,
)

x = torch.randn(4, 18)

y = model(x)

print("=" * 60)
print("Input shape :", x.shape)

print("=" * 60)
print("Output shape:", y.shape)