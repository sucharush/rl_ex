from src.models_old.agent_base import AgentBase
import numpy as np
from collections import deque
from typing import Optional
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.models.agent_ann_base import ANNAgentBase
import matplotlib.pyplot as plt
import wandb


class NStepAgentANN(ANNAgentBase):
    def __init__(
        self,
        nA: int,
        wandb=None,
        n_step=1,
        gamma=0.99,
        batch_size=64,
        target_update_freq: int = 100,
        trian_freq: int = 1,
        gradient_step: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(nA=nA, *args, **kwargs)
        self.n = n_step
        self.gamma = gamma
        self.buffer = deque()
        self.batch_size = batch_size
        self.train_freq = trian_freq  # do an update every 4 env steps
        self.gradient_steps = (
            gradient_step  # how many minibatch updates per training phase
        )
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.wandb = wandb
        self.target_tau

    def start_episode(self, s):
        self.buffer.clear()
        a = self.select_action(s)
        return a

    def step(self, s, a, r, s2, done, stalled=False):
        a2 = None if done else self.select_action(s2)
        if stalled and not done:
            a2 = self.fallback_action(s2)
        self.buffer.append((s, a, r))
        if len(self.buffer) >= self.n:
            self._update(s2, a2, done)
        return None if done else a2

    def step(self, s, a, r, s2, done, stalled=False):
        a2 = None if done else self.select_action(s2)
        if stalled and not done:
            a2 = self.fallback_action(s2)

        self.buffer.append((s, a, r))

        if len(self.buffer) >= self.n:
            self._process_n_step(s2, done)
        return None if done else a2


    def _update(self):
        s, a, G, s2, done = self.replay_buffer.sample(self.batch_size)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a), dtype=torch.long, device=self.device)
        G = torch.tensor(np.array(G), dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(done), dtype=torch.float32, device=self.device)
        # predicted Q(s,a)
        q_pred = self.Q_net(s).gather(1, a.unsqueeze(1)).squeeze()

        # loss + backprop
        loss = self.loss_fn(q_pred, G)
        self.optimizer.zero_grad()
        loss.backward()
        g_norm = self.grad_norm(self.Q_net.parameters())
        self.gradient_history.append(g_norm)
        torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), max_norm=30.0)
        self.optimizer.step()
        self.loss_history.append(loss.item())
        if self.wandb:
            self.wandb.log({"train/loss": loss.item(), "train/grad_norm": g_norm})
        if self.step_count % self.target_update_freq == 0:
            self.update_target()  # full copy

    def _process_n_step(self, s_next, done):
        """Turn local n-step buffer into one (s,a,G,s_next,done) transition."""
        s0, a0, _ = self.buffer[0]
        # compute discounted return
        G = 0.0
        for i, (_, _, r) in enumerate(self.buffer):
            G += (self.gamma**i) * r

        # bootstrap if not done
        if not done:
            s_next_tensor = torch.tensor(
                s_next, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_next = self.Q_target(s_next_tensor).max(1)[
                    0
                ]  # target net, not online
            G += (self.gamma**self.n) * q_next.item()

        # add to replay buffer
        self.replay_buffer.add((s0, a0, G, s_next, done))

        # drop oldest step
        self.buffer.popleft()

        # increment global step counter
        self.step_count += 1

        # only train every train_freq env steps
        if (
            self.step_count % self.train_freq == 0
            and len(self.replay_buffer) >= self.batch_size
        ):
            for _ in range(self.gradient_steps):
                self._update()
        # if len(self.replay_buffer) >= self.batch_size:
        #     self._update()

    def finish_episode(self, s_last):
        """Flush remaining buffer with shorter returns"""
        while self.buffer:
            self._update()

    def finish_episode(self, s_last):
        """Flush remaining buffer with shorter returns at episode end."""
        while self.buffer:
            s0, a0, _ = self.buffer[0]
            G = 0.0
            for i, (_, _, r) in enumerate(self.buffer):
                G += (self.gamma**i) * r
            # no bootstrap since done=True
            self.replay_buffer.add((s0, a0, G, s_last, True))
            self.buffer.popleft()
        # optionally, trigger some updates after flushing
        if len(self.replay_buffer) >= self.batch_size:
            for _ in range(self.gradient_steps):
                self._update()

    def compute_target(self, s_tensor, a):
        # On-policy SARSA target
        if a is None:
            return torch.tensor(0.0)
        return self.Q_net(s_tensor)[0, a]

    @staticmethod
    def grad_norm(parameters, norm_type=2):
        """Compute total norm of gradients across parameters."""
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        return total_norm ** (1.0 / norm_type)
