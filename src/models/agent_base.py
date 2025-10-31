
from abc import ABC, abstractmethod

class AgentBase(ABC):
    def __init__(self, nA: int):
        self.nA = nA
    @abstractmethod
    def q_values(self, s):
        """Returns the Q-values for each action in the given state."""
        pass
    # Q_net(s) for NNs, Q[s] for Tabular

    # ---- Fallback logic ----
    @abstractmethod
    def fallback_action(self, s):
        """Return a fallback action when agent is stuck."""
        pass

    # ---- Episode lifecycle ----
    @abstractmethod
    def step(self, s, a, r, s2, done, stalled):
        """
        Core update + action selection.
        Should return the next action to take.
        """
        pass

    def start_episode(self):
        """Optional setup at the beginning of an episode."""
        return None

    def finish_episode(self):
        """Optional cleanup/logging after an episode."""
        return None

    def update_params():
        pass