import argparse
import csv
import time
from pathlib import Path

import numpy as np

from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.config.config import load_config
from mappo_ss.envs.pettingzoo_env import PettingZooEnv


def main():

    # -------------------------------------------------
    # Parse command-line argument
    # -------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MAPPO model."
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to experiment directory.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print reward for every evaluation episode.",
    )

    args = parser.parse_args()

    experiment_dir = args.experiment

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------
    cfg = load_config(
        experiment_dir / "config.yaml"
    )

    # -------------------------------------------------
    # Create environment
    # -------------------------------------------------
    env = PettingZooEnv(cfg)

    env.reset()

    # -------------------------------------------------
    # Create agent
    # -------------------------------------------------
    agent = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=cfg,
    )

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    num_episodes = cfg["training"]["num_eval_episodes"]

    seed = cfg["training"]["seed"]

    episode_rewards = []

    print(f"Evaluating {num_episodes} episodes...")


    for episode in range(num_episodes):

        observations, _ = env.reset(
            seed=seed + episode
        )

        done = False

        episode_reward = 0.0

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

            # Remove if you don't want slow playback
            time.sleep(0.5)

            episode_reward += np.mean(
                list(rewards.values())
            )

            observations = next_observations

            done = (
                all(terminations.values())
                or all(truncations.values())
            )

        episode_rewards.append(
            episode_reward
        )

        if args.verbose:
            print(
                f"Episode {episode + 1}: "
                f"{episode_reward:.2f}"
            )
        elif (episode + 1) % 10 == 0:
            print(
                f"Evaluating... "
                f"{episode + 1}/{num_episodes} episodes completed"
            )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    best_reward = np.max(episode_rewards)
    worst_reward = np.min(episode_rewards)

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Average Reward : {avg_reward:.2f}")
    print(f"Std Reward     : {std_reward:.2f}")
    print(f"Best Reward    : {best_reward:.2f}")
    print(f"Worst Reward   : {worst_reward:.2f}")

    # -------------------------------------------------
    # Save results inside experiment folder
    # -------------------------------------------------
    csv_file = (
        experiment_dir
        / "evaluation_results.csv"
    )

    summary_file = (
        experiment_dir
        / "summary.txt"
    )


    file_exists = csv_file.exists()

    with open(
        csv_file,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "Experiment",
                "Checkpoint",
                "Episodes",
                "AverageReward",
                "StdReward",
                "BestReward",
                "WorstReward",
            ])

        writer.writerow([
            experiment_dir.name,
            checkpoint.name,
            num_episodes,
            avg_reward,
            std_reward,
            best_reward,
            worst_reward,
        ])

    # -------------------------------------------------
    # Save summary
    # -------------------------------------------------
    with open(summary_file, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("MAPPO Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Experiment : {experiment_dir.name}\n")
        f.write(f"Checkpoint : {checkpoint.name}\n")
        f.write(f"Episodes   : {num_episodes}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"Average Reward : {avg_reward:.2f}\n")
        f.write(f"Std Reward     : {std_reward:.2f}\n")
        f.write(f"Best Reward    : {best_reward:.2f}\n")
        f.write(f"Worst Reward   : {worst_reward:.2f}\n")
        f.write("-" * 60 + "\n")

    print()
    print(f"Results saved to : {csv_file}")
    print(f"Summary saved to : {summary_file}")

    env.close()


if __name__ == "__main__":
    main()