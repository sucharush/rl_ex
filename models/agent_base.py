from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

class AgentBase(ABC):
    def __init__(self, nA, alpha=0.1, gamma=1.0, eps=0.1, rng=None):
        self.nA = nA
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = rng or np.random.default_rng()
        self.Q = defaultdict(lambda: np.zeros(nA, dtype=float))

    def epsilon_greedy(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))
    
    @abstractmethod
    def start_episode(self, s):
        """reset any episode-specific buffers"""
        pass

    @abstractmethod
    def step(self, s, a, r, s2, done, stalled):
        """update Q-table and return next action"""
        pass

    @abstractmethod
    def fallback_action(self):
        pass

    