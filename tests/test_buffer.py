import numpy as np

from mappo_ss.buffers.rollout_buffer import RolloutBuffer
from mappo_ss.buffers.transition import Transition

buffer = RolloutBuffer()

transition = Transition(
    observations={
        "agent_0": np.zeros(18),
        "agent_1": np.zeros(18),
        "agent_2": np.zeros(18),
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

buffer.add(transition)

print("=" * 60)
print("Buffer size:", len(buffer))

print("=" * 60)
print(buffer.transitions[0])