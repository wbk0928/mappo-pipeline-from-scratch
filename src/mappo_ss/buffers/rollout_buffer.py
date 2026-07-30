"""
Rollout buffer for MAPPO.
"""

from __future__ import annotations

from mappo_ss.buffers.transition import Transition


class RolloutBuffer:
    """
    Stores a rollout consisting of many transitions.
    """

    def __init__(self):

        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        """
        Store one transition.
        """

        self.transitions.append(transition)

    def clear(self) -> None:
        """
        Empty the rollout buffer.
        """

        self.transitions.clear()

    def __len__(self) -> int:

        return len(self.transitions)