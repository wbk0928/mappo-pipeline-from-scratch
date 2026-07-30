from __future__ import annotations

from typing import Dict

import torch
import numpy as np
from mappo_ss.algorithms.running_mean_std import RunningMeanStd
from mappo_ss.algorithms.value_norm import ValueNorm
from mappo_ss.algorithms.ppo import PPOUpdater
from mappo_ss.buffers.rollout_buffer import RolloutBuffer
from mappo_ss.networks.actor import Actor
from mappo_ss.networks.critic import Critic
from mappo_ss.algorithms.gae import compute_gae

class MAPPO:

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        cfg,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)

        hidden_dim = cfg["network"]["actor_hidden_dim"]

        self.actor = Actor(
            obs_dim,
            hidden_dim,
            action_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim,
            cfg["network"]["critic_hidden_dim"],
        ).to(self.device)

        self.obs_normalizer = RunningMeanStd(
            shape=obs_dim,
            device=self.device,
        )

        self.value_normalizer = ValueNorm()

        lr = cfg["algorithm"]["learning_rate"]

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

        self.buffer = RolloutBuffer()

        self.ppo = PPOUpdater(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            value_normalizer=self.value_normalizer,

            clip_ratio=cfg["algorithm"]["clip_ratio"],
            value_coef=cfg["algorithm"]["value_coef"],
            entropy_coef=cfg["algorithm"]["entropy_coef"],
            max_grad_norm=cfg["algorithm"]["max_grad_norm"],
        )

    @torch.no_grad()
    def act(
        self,
        observations: Dict[str, torch.Tensor],
    ):
        """
        Select actions for every agent.
        """

        actions = {}
        log_probs = {}

        for agent_id, obs in observations.items():

            obs = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Update running statistics
            self.obs_normalizer.update(obs)

            # Normalize observation
            obs = self.obs_normalizer.normalize(obs)

            dist = self.actor(obs)

            action = dist.sample()

            actions[agent_id] = int(action.item())

            log_probs[agent_id] = float(
                dist.log_prob(action).item()
            )

        return actions, log_probs

    @torch.no_grad()
    def act_deterministic(
    self,
    observations,
    ):
        """
        Select greedy actions for evaluation.
        """

        actions = {}

        for agent, obs in observations.items():

            obs = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

             #Do NOT update statistics during evaluation
            obs = self.obs_normalizer.normalize(obs)

            dist = self.actor(obs)

            action = torch.argmax(
                dist.probs,
                dim=-1,
            )

            actions[agent] = int(action.item())

        return actions
    
    def evaluate(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        global_states: torch.Tensor,
    ):
        """
        Evaluate stored actions.

        Used during PPO updates.
        """

        dist = self.actor(observations)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        values = self.critic(global_states)

        return log_probs, entropy, values

    def store_transition(
        self,
        transition,
    ):
        """
        Store one environment transition.
        """

        self.buffer.add(transition)

    def ready(self) -> bool:
        """
        Returns True when enough rollout data
        has been collected.
        """

        rollout_length = self.cfg["training"]["rollout_length"]

        return len(self.buffer) >= rollout_length

    def _create_minibatches(
        self,
        num_samples: int,
    ):
        """
        Yield shuffled minibatch indices.
        """

        minibatch_size = self.cfg["algorithm"]["minibatch_size"]

        indices = torch.randperm(
            num_samples,
            device=self.device,
        )

        for start in range(
            0,
            num_samples,
            minibatch_size,
        ):

            end = start + minibatch_size

            yield indices[start:end]

    def _prepare_training_batch(self):
        """
        Convert the rollout buffer into one flattened PPO batch.
        """
        transitions = self.buffer.transitions
        
        
        
        # -----------------------------
        # Step 2: Get agent IDs
        # -----------------------------
        agent_ids = sorted(
            transitions[0].observations.keys()
        )
        
        # -----------------------------
        # Step 3: Create rollout storage
        # -----------------------------
        rollout = {}
        
        for agent in agent_ids:
        
            rollout[agent] = {
                "observations": [],
                "global_states": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "values": [],
                "dones": [],
                        
            }
        
        # -----------------------------
        # Step 4: Fill rollout
        # -----------------------------
        for transition in transitions:
        
            for agent in agent_ids:
        
                rollout[agent]["observations"].append(
                    transition.observations[agent]
        
                )
        
                rollout[agent]["global_states"].append(
                    transition.global_state
                )
        
        
        
                rollout[agent]["actions"].append(
                    transition.actions[agent]
                        )
        
                rollout[agent]["log_probs"].append(
                    transition.log_probs[agent]
                        )
        
                rollout[agent]["rewards"].append(
                    transition.rewards[agent]
                        )
        
                rollout[agent]["values"].append(
                    transition.values[agent]
                )
        
                done = (
                    transition.terminations[agent]
                    or transition.truncations[agent]
                )
        
                rollout[agent]["dones"].append(done)
        
        # -----------------------------
        # Step 5: Convert to tensors
        # -----------------------------
        for agent in agent_ids:
        
            rollout[agent]["observations"] = torch.as_tensor(
                        np.asarray(rollout[agent]["observations"]),
                        dtype=torch.float32,
                        device=self.device,
                    )
        
            rollout[agent]["global_states"] = torch.as_tensor(
                        np.asarray(
                        rollout[agent]["global_states"]
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
        
            rollout[agent]["actions"] = torch.as_tensor(
                        rollout[agent]["actions"],
                        dtype=torch.long,
                        device=self.device,
                    )
        
            rollout[agent]["log_probs"] = torch.as_tensor(
                        rollout[agent]["log_probs"],
                        dtype=torch.float32,
                        device=self.device,
                    )
        
            rollout[agent]["rewards"] = torch.as_tensor(
                        rollout[agent]["rewards"],
                        dtype=torch.float32,
                        device=self.device,
                    )
        
            rollout[agent]["values"] = torch.as_tensor(
                        rollout[agent]["values"],
                        dtype=torch.float32,
                        device=self.device,
                    )
        
            rollout[agent]["dones"] = torch.as_tensor(
                        rollout[agent]["dones"],
                        dtype=torch.float32,
                        device=self.device,
                    )
        
                    
        # -----------------------------
        # Step 6: Compute GAE
        # -----------------------------
        for agent in agent_ids:

            last_transition = transitions[-1]

            done = (
                last_transition.terminations[agent]
                or last_transition.truncations[agent]
            )

            if done:

                next_value = torch.tensor(
                0.0,
                device=self.device,
            )

            else:

                next_state = torch.as_tensor(
                last_transition.next_global_state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                with torch.no_grad():

                    next_value = self.critic(
                    next_state
                    ).squeeze(0)

            advantages, returns = compute_gae(
                rewards=rollout[agent]["rewards"],
                values=rollout[agent]["values"],
                dones=rollout[agent]["dones"],
                next_value=next_value,
                gamma=self.cfg["algorithm"]["gamma"],
                gae_lambda=self.cfg["algorithm"]["gae_lambda"],
    )

            rollout[agent]["advantages"] = advantages
            rollout[agent]["returns"] = returns
            
        
        # Step 7: Normalize advantages
        # across all agents
        # -----------------------------

        all_advantages = torch.cat(
            [
                rollout[a]["advantages"]
                for a in agent_ids
            ],
            dim=0,
        )

        all_advantages = (
            all_advantages
            - all_advantages.mean()
        ) / (
            all_advantages.std() + 1e-8
        )

        start = 0

        for agent in agent_ids:

            length = rollout[agent]["advantages"].shape[0]

            rollout[agent]["advantages"] = (
            all_advantages[start:start + length]
            )

            start += length
        
        # -----------------------------
        # Step 8: Flatten trajectories
        # -----------------------------
        
        observations = torch.cat(
                    [rollout[a]["observations"] for a in agent_ids],
                    dim=0,
                )

        with torch.no_grad():
            observations = self.obs_normalizer.normalize(
            observations
        )
        
        global_states = torch.cat(
                    [rollout[a]["global_states"] for a in agent_ids],
                    dim=0,
                )
        
        actions = torch.cat(
                    [rollout[a]["actions"] for a in agent_ids],
                    dim=0,
                )
        
        old_log_probs = torch.cat(
                    [rollout[a]["log_probs"] for a in agent_ids],
                    dim=0,
                )
        values = torch.cat(
                    [rollout[a]["values"] for a in agent_ids],
                    dim=0,
                )
        old_values = self.value_normalizer.normalize(values)
        
        advantages = torch.cat(
                    [rollout[a]["advantages"] for a in agent_ids],
                    dim=0,
                )
        
        returns = torch.cat(
                    [rollout[a]["returns"] for a in agent_ids],
                    dim=0,
                )
        return (
            observations,
            global_states,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
        )  
    

    def learn(self):
        """
        Perform one MAPPO update.
        """

        if len(self.buffer) == 0:
            return None

        if not self.ready():
            return None

        (
            observations,
            global_states,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
        ) = self._prepare_training_batch()


        # Update running statistics ONCE per rollout
        self.value_normalizer.update(returns)

        policy_loss = 0.0
        critic_loss = 0.0
        entropy = 0.0

        num_updates = 0

        ppo_epochs = self.cfg["algorithm"]["ppo_epochs"]

        for _ in range(ppo_epochs):

            for batch_idx in self._create_minibatches(
                len(observations)
            ):

                batch_observations = observations[batch_idx]

                batch_global_states = global_states[batch_idx]

                batch_actions = actions[batch_idx]

                batch_old_log_probs = old_log_probs[batch_idx]
                batch_old_values = old_values[batch_idx]

                batch_advantages = advantages[batch_idx]

                batch_returns = returns[batch_idx]

                stats = self.ppo.update(
                    observations=batch_observations,
                    global_states=batch_global_states,
                    actions=batch_actions,
                    old_log_probs=batch_old_log_probs,
                    old_values=batch_old_values,
                    advantages=batch_advantages,
                    returns=batch_returns,
                )

                policy_loss += stats["policy_loss"]
                critic_loss += stats["critic_loss"]
                entropy += stats["entropy"]

                num_updates += 1

        stats = {
            "policy_loss": policy_loss / num_updates,
            "critic_loss": critic_loss / num_updates,
            "entropy": entropy / num_updates,
        }

        self.buffer.clear()

        return stats


    def update_learning_rate(
        self,
        current_episode: int,
        total_episodes: int,
    ):
        """
        Linearly decay the learning rate.
        """

        initial_lr = self.cfg["algorithm"]["learning_rate"]

        lr = initial_lr * (
            1.0
            - current_episode / total_episodes
        )

        lr = max(lr, 0.0)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = lr
       

    def save(
        self,
        path,
    ):
        """
        Save the complete MAPPO checkpoint.
        """

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_norm": self.obs_normalizer.state_dict(),
                "value_norm": self.value_normalizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(
        self,
        path,
    ):
        """
        Load a MAPPO checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.obs_normalizer.load_state_dict(
            checkpoint["obs_norm"]
        )

        self.value_normalizer.load_state_dict(
            checkpoint["value_norm"]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )
       
    
        

        