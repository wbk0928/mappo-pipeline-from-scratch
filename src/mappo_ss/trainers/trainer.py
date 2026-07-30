"""
MAPPO Trainer.
"""

from __future__ import annotations


class Trainer:

    def __init__(
        self,
        env,
        agent,
        cfg,
    ):

        self.env = env
        self.agent = agent
        self.cfg = cfg

    def train(self):

        total_episodes = self.cfg["training"]["total_episodes"]

        for episode in range(total_episodes):

            observations, global_state = self.env.reset()

            episode_reward = 0.0

            done = False

            while not done:

                actions, log_probs = self.agent.act(
                observations
                )

                (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
                next_global_state,
                ) = self.env.step(actions)

                done = (
                all(terminations.values())
                or all(truncations.values())
                )

                observations = next_observations

                global_state = next_global_state

                episode_reward += sum(
                rewards.values()
                )

            print(
                f"Episode {episode:5d}"
                f" | Reward {episode_reward:.2f}"
            )