"""
Test MAPPO.learn().
"""

import numpy as np
print("Starting test_learn...")

from mappo_ss.algorithms.mappo import MAPPO
from mappo_ss.buffers.transition import Transition
from mappo_ss.config.config import load_config


# -----------------------
# Create MAPPO agent
# -----------------------

cfg = load_config("configs/simple_spread.yaml")

agent = MAPPO(
    obs_dim=18,
    state_dim=54,
    action_dim=5,
    cfg=cfg,
)

# -----------------------
# Create fake rollout
# -----------------------

for step in range(4):

    transition = Transition(

        observations={
            "agent_0": np.zeros(18),
            "agent_1": np.ones(18),
            "agent_2": np.full(18, 2),
        },

        global_state=np.zeros(54),

        actions={
            "agent_0": 0,
            "agent_1": 1,
            "agent_2": 2,
        },

        log_probs={
            "agent_0": -1.5,
            "agent_1": -1.4,
            "agent_2": -1.6,
        },

        rewards={
            "agent_0": 1.0,
            "agent_1": 1.0,
            "agent_2": 1.0,
        },

        values={
            "agent_0": 0.5,
            "agent_1": 0.5,
            "agent_2": 0.5,
        },

        next_observations={
            "agent_0": np.zeros(18),
            "agent_1": np.ones(18),
            "agent_2": np.full(18, 2),
        },

        next_global_state=np.zeros(54),

        terminations={
            "agent_0": False,
            "agent_1": False,
            "agent_2": False,
        },

        truncations={
            "agent_0": False,
            "agent_1": False,
            "agent_2": False,
        },
    )

    agent.store_transition(transition)

# -----------------------
# Call learn()
# -----------------------

agent.learn()