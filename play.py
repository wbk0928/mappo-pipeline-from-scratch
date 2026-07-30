"""
Play a trained MAPPO agent.

Loads the best checkpoint from an experiment directory and
visualizes the learned policy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.config.config import load_config
from mappo_ss.envs.pettingzoo_env import PettingZooEnv


def main():

    # --------------------------------------------------
    # Parse command-line argument
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Play a trained MAPPO agent."
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to experiment directory.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5).",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay (seconds) between environment steps.",
    )

    args = parser.parse_args()

    experiment_dir = args.experiment

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    cfg = load_config(
        experiment_dir / "config.yaml"
    )

    # Force rendering for play
    cfg["environment"]["render_mode"] = "human"

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------

    env = PettingZooEnv(cfg)

    env.reset()

    # --------------------------------------------------
    # Create agent
    # --------------------------------------------------

    agent = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=cfg,
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = (
        experiment_dir
        / "checkpoints"
        / "best_model.pt"
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    print(f"Loading checkpoint: {checkpoint}")

    agent.load(checkpoint)

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------
    # Play episodes
    # --------------------------------------------------
    
    num_episodes = args.episodes
    seed = cfg["training"]["seed"]

    episode_rewards = []
    print(f"\nPlaying {num_episodes} episode(s)...")

    for episode in range(num_episodes):

        observations, _ = env.reset(
            seed=seed + episode
        )

        done = False

        episode_reward = 0.0

        print("=" * 60)
        print(f"Episode {episode + 1}")

        while not done:

            actions = agent.act_deterministic(
                observations
            )

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            episode_reward += np.mean(
                list(rewards.values())
            )

            observations = next_observations

            done = (
                all(terminations.values())
                or all(truncations.values())
            )

            time.sleep(args.delay)

        episode_rewards.append(
            episode_reward
        )

        print(
            f"Reward: {episode_reward:.2f}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Play Summary")
    print("=" * 60)

    print(
        f"Average Reward : "
        f"{np.mean(episode_rewards):.2f}"
    )

    print(
        f"Best Reward    : "
        f"{np.max(episode_rewards):.2f}"
    )

    print(
        f"Worst Reward   : "
        f"{np.min(episode_rewards):.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()