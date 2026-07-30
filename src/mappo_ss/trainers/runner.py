"""
Runner for MAPPO training.
"""

from __future__ import annotations
import torch
import shutil
from datetime import datetime
import numpy as np
from mappo_ss.utils.logger import get_logger
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from mappo_ss.buffers.transition import Transition


class Runner:
    """
    Collects experience from the environment
    and trains the MAPPO agent.
    """

    def __init__(
        self,
        env,
        agent,
        cfg,
    ):

        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.debug = self.cfg["logging"].get(
        "debug",
        False,
        )

        self.best_reward = float("-inf")

        self.logger = get_logger(__name__)

        # -------------------------------------------------
        # Experiment information
        # -------------------------------------------------
        env_name = self.cfg["environment"]["env_name"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        algo = self.cfg["algorithm"]
        train = self.cfg["training"]

        run_name = (
            f"lr{algo['learning_rate']}_"
            f"clip{algo['clip_ratio']}_"
            f"ent{algo['entropy_coef']}_"
            f"ep{algo['ppo_epochs']}_"
            f"mb{algo['minibatch_size']}_"
            f"train{train['total_episodes']}_"
            f"seed{train['seed']}_"
            f"{timestamp}"
        )

        # -------------------------------------------------
        # Experiment directory
        # -------------------------------------------------
        self.experiment_dir = (
            Path("experiments")
            / env_name
            / run_name
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------
        # Checkpoints
        # -------------------------------------------------
        self.checkpoint_dir = (
            self.experiment_dir
            / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            exist_ok=True,
        )

        # -------------------------------------------------
        # TensorBoard
        # -------------------------------------------------
        tensorboard_dir = (
            self.experiment_dir
            / "tensorboard"
        )

        tensorboard_dir.mkdir(
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            tensorboard_dir
        )

        # -------------------------------------------------
        # Save configuration
        # -------------------------------------------------
        shutil.copy(
            "configs/simple_spread.yaml",
            self.experiment_dir / "config.yaml",
        )

        self.logger.info(
            f"Experiment directory: {self.experiment_dir}"
        )
        
    def run_episode(self):
        """
        Run one complete episode.
        """
        observations, _ = self.env.reset()

        if self.debug:
            self.logger.info("=" * 60)
            self.logger.info("Episode started")

        rollout_length = self.cfg["training"]["rollout_length"]

        episode_reward = 0.0

        for step in range(rollout_length):

            if self.debug:
                print("=" * 60)
                print(f"Step {step + 1}/{rollout_length}")
            

            global_state = self.env.get_global_state(
            observations
            )

            state_tensor = torch.as_tensor(
                global_state,
                dtype=torch.float32,
                device=self.agent.device,
            ).unsqueeze(0)

            value = self.agent.critic(state_tensor)
            if self.debug:
                print("=" * 60)
                print("Critic value")
                print(value.item())
            values = {
                agent: float(value.item())
                for agent in observations
            }

            actions, log_probs = self.agent.act(observations)

            if self.debug:       
                print("=" * 60)
                print("Selected actions")
                print(actions)

            if self.debug:
                print("=" * 60)
                print("Log probabilities")
                print(log_probs)  
         
            next_observations, rewards, terminations, truncations, infos = (
                self.env.step(actions)
            )   

            episode_reward += np.mean(
                list(rewards.values())
            )

            next_global_state = self.env.get_global_state(
                next_observations
            )   
            if self.debug:
                print("=" * 60)
                print("Rewards")
                print(rewards)

            if self.debug:
                print("=" * 60)
                print("Terminations")
                print(terminations)
                
            if self.debug:
                print("=" * 60)
                print("Truncations")
                print(truncations)

            transition = Transition(
            observations=observations,
            global_state=global_state,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            next_observations=next_observations,
            next_global_state=next_global_state,
            terminations=terminations,
            truncations=truncations,
            )

            self.agent.buffer.add(transition)

            if self.debug:
                self.logger.info("=" * 60)
                self.logger.info(
                    f"Buffer size: {len(self.agent.buffer)}"
                )

            observations = next_observations

        if self.debug:
            print("=" * 60)
            print("Rollout collected")
            print("Buffer size:", len(self.agent.buffer))

        stats = None
        if self.agent.ready():

            if self.debug:
                print("=" * 60)
                print("Starting PPO update")

            stats = self.agent.learn()

            if self.debug:
                
                print("Buffer cleared")
                print("Buffer size:", len(self.agent.buffer))

        return episode_reward, stats

    def evaluate(self):
        """
        Evaluate the current policy using deterministic actions.
        """

        num_eval_episodes = self.cfg["training"]["num_eval_episodes"]

        total_reward = 0.0

        for _ in range(num_eval_episodes):

            observations, _ = self.env.reset()

            done = False

            episode_reward = 0.0

            while not done:

                actions = self.agent.act_deterministic(
                    observations
                )

                (
                    next_observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = self.env.step(actions)

                episode_reward += np.mean(
                    list(rewards.values())
                )

                observations = next_observations

                done = (
                    all(terminations.values())
                    or all(truncations.values())
                )

            total_reward += episode_reward

        return total_reward / num_eval_episodes
   

    def train(self):
        """
        Train the MAPPO agent.
        """

        total_episodes = self.cfg["training"]["total_episodes"]

        save_every = self.cfg["logging"]["save_every"]
        eval_every = self.cfg["logging"]["eval_every"]

        for episode in range(total_episodes):

            # ---------------------------------
            # Learning-rate decay
            # ---------------------------------
            self.agent.update_learning_rate(
                current_episode=episode,
                total_episodes=total_episodes,
            )

            # ---------------------------------
            # Collect rollout + PPO update
            # ---------------------------------
            episode_reward, stats = self.run_episode()

            # ---------------------------------
            # TensorBoard
            # ---------------------------------
            self.writer.add_scalar(
                "Reward/Train",
                episode_reward,
                episode,
            )

            self.writer.add_scalar(
                "LearningRate",
                self.agent.actor_optimizer.param_groups[0]["lr"],
                episode,
            )

            if stats is not None:

                self.writer.add_scalar(
                    "Loss/Policy",
                    stats["policy_loss"],
                    episode,
                )

                self.writer.add_scalar(
                    "Loss/Critic",
                    stats["critic_loss"],
                    episode,
                )

                self.writer.add_scalar(
                    "Loss/Entropy",
                    stats["entropy"],
                    episode,
                )

            # ---------------------------------
            # Evaluation
            # ---------------------------------
            evaluation_reward = None

            if (episode + 1) % eval_every == 0:

                evaluation_reward = self.evaluate()

                self.writer.add_scalar(
                    "Reward/Evaluation",
                    evaluation_reward,
                    episode,
                )

                if evaluation_reward > self.best_reward:

                    self.best_reward = evaluation_reward

                    best_path = (
                        self.checkpoint_dir
                        / "best_model.pt"
                    )

                    self.agent.save(best_path)

            # ---------------------------------
            # Save checkpoint
            # ---------------------------------
            checkpoint_saved = False

            if (episode + 1) % save_every == 0:

                checkpoint_path = (
                    self.checkpoint_dir
                    / f"checkpoint_{episode + 1}.pt"
                )

                self.agent.save(checkpoint_path)

                checkpoint_saved = True

            # ---------------------------------
            # Console summary
            # ---------------------------------
            print("=" * 60)

            print(
                f"Episode {episode + 1}/{total_episodes}"
            )

            print()

            print(
                f"Train Reward : {episode_reward:.2f}"
            )

            if evaluation_reward is not None:

                print(
                    f"Eval Reward  : {evaluation_reward:.2f}"
                )

                print(
                    f"Best Reward  : {self.best_reward:.2f}"
                )

            print()

            if stats is not None:

                print(
                    f"Policy Loss  : {stats['policy_loss']:.6f}"
                )

                print(
                    f"Critic Loss  : {stats['critic_loss']:.6f}"
                )

                print(
                    f"Entropy      : {stats['entropy']:.6f}"
                )

            print()

            print(
                "LR           : "
                f"{self.agent.actor_optimizer.param_groups[0]['lr']:.2e}"
            )

            print()

            print(
                "Checkpoint   : "
                + (
                    "Saved"
                    if checkpoint_saved
                    else "Not Saved"
                )
            )

            print("-" * 60)

        self.writer.close()