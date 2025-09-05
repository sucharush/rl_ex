from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from src.utils.scheduler import ScheduleParam
from typing import Optional


class AgentBase(ABC):
    def __init__(
        self,
        nA,
        alpha: ScheduleParam,
        eps: ScheduleParam,
        c: ScheduleParam,
        gamma: float = 1.0,
        policy: str = "egreedy",
        rng: Optional[np.random.Generator] = None,
    ):
        self.nA = nA
        self.policy = policy
        self.alpha, self.gamma, self.eps, self.c = alpha, gamma, eps, c
        self.rng = rng or np.random.default_rng()
        self.Q = defaultdict(lambda: np.zeros(nA, dtype=float))
        self.N = defaultdict(lambda: np.zeros(nA, dtype=int))  # action visit counts

        # choose once, no repeated branching
        if policy == "egreedy":
            self.select_action = self.epsilon_greedy
        elif policy == "ucb":
            self.select_action = self.ucb_action
        else:
            raise ValueError(f"Unknown policy: {policy}")

    @staticmethod
    def choose_top(q_values):
        best = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(best))

    def update_params(self):
        self.alpha.step()
        self.eps.step()
        self.c.step()

    def epsilon_greedy(self, s):
        if self.rng.random() < self.eps():
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        return self.choose_top(q)

    def start_episode(self, s):
        a = self.select_action(s)
        if self.policy == "ucb":
            self.N[s][a] += 1
        return a

    def ucb_action(self, s):
        n = self.N[s].astype(float)  # ensure float division
        total = np.sum(n) + 1  # total visits of state s

        q = self.Q[s]

        # compute bonus safely
        bonus = np.zeros_like(q, dtype=float)
        valid = n > 0
        np.divide(np.log(total), n, out=bonus, where=valid)
        ucb_values = q + self.c() * np.sqrt(bonus)

        # force exploration for unvisited actions only
        ucb_values[~valid] = np.inf
        a = self.choose_top(ucb_values)
        self.N[s][a] += 1
        return a

    def fallback_action(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        all_actions = np.arange(self.nA)

        # mask out best ones
        candidates = np.setdiff1d(all_actions, best)

        if len(candidates) == 0:
            # if all actions tie, just pick random
            return int(self.rng.integers(self.nA))

        return int(self.rng.choice(candidates))

    @abstractmethod
    def step(self, s, a, r, s2, done, stalled):
        """update Q-table and return next action"""
        pass
