from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.config.config import load_config

cfg = load_config("configs/simple_spread.yaml")

agent = MAPPO(
    obs_dim=18,
    state_dim=54,
    action_dim=5,
    cfg=cfg,
)

print("=" * 60)
print("Ready initially:")
print(agent.ready())