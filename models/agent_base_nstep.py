from models.agent_base import AgentBase
import numpy as np
from collections import deque
from abc import abstractmethod


class NStepAgentBase(AgentBase):
    def __init__(self, nA, n_step=1, alpha=0.1, gamma=1.0, eps=0.1, rng=None):
        super().__init__(nA, alpha, gamma, eps, rng)
        self.n = n_step
        self.buffer = deque()

    def fallback_action(self):
        return self.nA - 1

    def start_episode(self, s):
        self.buffer.clear()
        return self.epsilon_greedy(s)

    def step(self, s, a, r, s2, done, stalled=False):
        if stalled:
            a = self.fallback_action()

        a2 = None if done else self.epsilon_greedy(s2)
        # store only what happened at this step
        self.buffer.append((s, a, r))

        if len(self.buffer) >= self.n:
            self._update(s2, a2, done)

        return None if done else a2

    def _update(self, s_next, a_next, done):
        # unpack oldest transition
        s0, a0, _ = self.buffer[0]

        # compute n-step return
        G = 0.0
        for i, (_, _, r) in enumerate(self.buffer):
            G += (self.gamma ** i) * r

        if not done:
            G += (self.gamma ** self.n) * self.compute_target(s_next, a_next)

        # TD update
        self.Q[s0][a0] += self.alpha * (G - self.Q[s0][a0])

        # drop oldest
        self.buffer.popleft()

    def finish_episode(self, s_last):
        """Flush remaining buffer with shorter returns"""
        while self.buffer:
            self._update(s_last, None, done=True)

    @abstractmethod
    def compute_target(self, s, a):
        pass
