from mappo_ss.config.config import load_config
from mappo_ss.envs.pettingzoo_env import PettingZooEnv
from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.trainers.runner import Runner
from mappo_ss.utils.seeding import set_seed


def main():

    cfg = load_config("configs/simple_spread.yaml")

    seed = cfg["training"]["seed"]
    set_seed(seed)

    env = PettingZooEnv(cfg)

    # Initialize environment and dimensions
    env.reset(seed=seed)

    agent = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=cfg,
    )

    runner = Runner(
        env,
        agent,
        cfg,
    )

    runner.train()

    env.close()


if __name__ == "__main__":
    main()