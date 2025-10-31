from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from src.utils.scheduler import ScheduleParam
from src.models.agent_base import AgentBase
from typing import Optional
from collections import deque
import random

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=None):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            # Pure linear model
            self.layers = nn.Linear(state_dim, n_actions)
        else:
            self.layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)

class ANNAgentBase(AgentBase):
    def __init__(
        self,
        nA: int,
        alpha: ScheduleParam,
        eps: ScheduleParam,
        network: nn.Module,
        target_tau: float = 0.005,
        policy: str = "egreedy",
        loss_fn: nn.Module = nn.MSELoss(),
        device: str = "cpu",
        debug: bool = False,
    ):
        super().__init__(nA=nA)
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.policy = policy
        self.alpha, self.eps = alpha, eps
        self.device = torch.device(device)
        self.debug = debug
        # # ANN replaces Q-table
        # self.loss_fn = nn.MSELoss()
        self.Q_net = network.to(self.device)
        self.Q_target = copy.deepcopy(self.Q_net)
        # self.optimizer = optimizer or optim.Adam(self.Q_net.parameters(), lr=(self.alpha()))
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=float(self.alpha()))

        self.loss_fn = loss_fn
        # Policy selection
        if policy == "egreedy":
            self.select_action = self.epsilon_greedy
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if self.debug:
            self.debug_print = lambda *args, **kwargs: print(*args, **kwargs)
        else:
            self.debug_print = lambda *args, **kwargs: None
        self.target_tau = target_tau
        self.loss_history = []
        self.gradient_history = []

    def q_values(self, s: torch.Tensor) -> torch.Tensor:
        """Return Q(s, :) as a 1D tensor (on correct device)."""
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        s = s.to(self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)  # batch of 1
        with torch.no_grad():
            q = self.Q_net(s)  # shape (1, nA)
        return q.squeeze(0)  # shape (nA,)

    def update_target(self, hard=False):
        """Update target net: hard copy or Polyak averaging."""
        if hard:
            self.Q_target.load_state_dict(self.Q_net.state_dict())
        else:
            for p, p_targ in zip(self.Q_net.parameters(), self.Q_target.parameters()):
                p_targ.data.copy_(self.target_tau * p.data + (1 - self.target_tau) * p_targ.data)

    def update_params(self):
        self.eps.step()
        self.alpha.step()

    def debug_mode(self):
        self.debug = True
        self.debug_print = lambda *args, **kwargs: print(*args, **kwargs)

    @staticmethod
    def choose_top(q_values: torch.Tensor) -> int:
        """Break ties randomly among max values."""
        max_val = q_values.max()
        best = torch.nonzero(q_values == max_val, as_tuple=False).squeeze()
        if best.ndim == 0:
            return int(best.item())
        idx = torch.randint(len(best), (1,))
        return int(best[idx].item())

    def epsilon_greedy(self, s):
        if torch.rand(1).item() < self.eps():
            return int(torch.randint(self.nA, (1,)))
        q = self.q_values(s)
        self.debug_print(f"[ε-greedy] q = {q}")
        return self.choose_top(q)

    def fallback_action(self, s):
        q = self.q_values(s)
        max_val = q.max()
        best = torch.nonzero(q == max_val, as_tuple=False).squeeze().tolist()
        if isinstance(best, int):
            best = [best]
        all_actions = set(range(self.nA))
        candidates = list(all_actions - set(best))
        if not candidates:
            return int(torch.randint(self.nA, (1,)))
        idx = torch.randint(len(candidates), (1,))
        return candidates[idx]

    def step(self, s, a, r, s2, done, stalled):
        # (network update logic)
        # must return next action
        raise NotImplementedError("Implement update + next action selection")

    def plot_loss(self, window: int = 100, xlim=None, ylim=None):
        if not self.loss_history:
            print("No loss recorded.")
            return

        losses = np.array(self.loss_history)

        # Compute moving average with convolution
        if len(losses) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(losses, kernel, mode="valid")
        else:
            smooth = losses  # not enough points to smooth

        plt.plot(losses, alpha=0.3, label="Raw loss")  # faint raw curve
        plt.plot(
            range(window - 1, window - 1 + len(smooth)),
            smooth,
            label=f"Smoothed (window={window})",
        )
        if ylim:
            plt.ylim([0, ylim])
        plt.xlabel("Update step")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.legend()
        plt.show()

    def replay_buffer(self):
        pass
