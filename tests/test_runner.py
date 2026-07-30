from mappo_ss.config.config import load_config
from mappo_ss.envs.pettingzoo_env import PettingZooEnv
from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.trainers.runner import Runner

cfg = load_config("configs/simple_spread.yaml")

env = PettingZooEnv(cfg)

env.reset()

agent = MAPPO(
    obs_dim=env.obs_dim,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    cfg=cfg,
)

runner = Runner(
    env=env,
    agent=agent,
    cfg=cfg,
)

runner.run_episode()