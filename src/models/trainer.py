from src.environments.environment import RectangleEnv
import numpy as np
from collections import defaultdict
from src.models.agent_ann_base import ANNAgentBase
from typing import Dict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch


class Trainer:
    def __init__(
        self,
        envs: Dict[str, RectangleEnv] | list[RectangleEnv] | RectangleEnv | callable,
        agent: ANNAgentBase,
        reuse_per_env: int = 1,
    ):
        """
        Either pass:
          - envs: dict/list/RectangleEnv (fixed pool), OR
          - env_generator: callable that returns a RectangleEnv
        """
        self.agent = agent
        self.reuse_per_env = reuse_per_env
        # self.loss_history = []

        if callable(envs):  # env generator
            self.envs = envs
            self.is_generator = True
        else:
            if isinstance(envs, dict):
                self.envs = envs
            elif isinstance(envs, list):
                self.envs = {f"env{i}": e for i, e in enumerate(envs)}
            elif isinstance(envs, RectangleEnv):
                self.envs = {"default": envs}
            self.is_generator = False

    def _get_env(self, ep):
        if self.is_generator:
            if ep % self.reuse_per_env == 0:
                self.current_env = self.envs()  # call generator
            return self.current_env
        else:
            env_names = list(self.envs.keys())
            return self.envs[env_names[(ep // self.reuse_per_env) % len(env_names)]]

    def fit(self, episodes=1000, log_interval=10):
        # self.agent.Q_net.eval()
        # self.agent.loss_history = []
        self.logs = []
        # env_names = list(self.envs.keys()) if not self.is_generator else None
        for ep in range(episodes):
            env = self._get_env(ep)  # get fresh or reuse
            s = env.reset()
            self.agent.update_params()
            a = self.agent.start_episode(s)
            done, G, steps = False, 0.0, 0
            while not done:
                s2, r, done = env.step(a)
                G += r
                steps += 1
                stalled = env.is_stalled()
                # print("[Trainer.fit]:", "Rect", env.rectangle.get_info(), "Stalled", stalled)
                a = self.agent.step(s, a, r, s2, done, stalled)
                # print("action",a)
                s = s2

            if hasattr(self.agent, "finish_episode"):
                self.agent.finish_episode(s)

            if ep % log_interval == 0:
                print(f"[Trainer.fit] Episode {ep+1}/{episodes}")
                self.logs.append(
                    {
                        "ep": ep,
                        "reward": G,
                        "steps": steps,
                        "n_iters": env.optimizer_iters,
                    }
                )

    def greedy_action(self, s):
        # q = self.agent.Q[s]
        q = self.agent.q_values(s)
        max_val = q.max()
        best = torch.nonzero(q == max_val, as_tuple=False).squeeze()
        if best.ndim == 0:
            return int(best.item())
        idx = torch.randint(len(best), (1,))
        return int(best[idx].item())

    def greedy_action(self, s):
        """Select greedy action (ties broken randomly), matching Q_net device."""
        device = self.agent.device  # detect model's device

        # Ensure input is a tensor on the same device
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32, device=device)
        else:
            s = s.to(device=device, dtype=torch.float32)

        # Compute Q(s, :)
        q = self.agent.q_values(s)

        # Find all indices with the max Q-value
        max_val = q.max()
        best = torch.nonzero(q == max_val, as_tuple=False).squeeze()

        # Break ties randomly on the same device
        if best.ndim == 0:
            action = best.item()
        else:
            idx = torch.randint(len(best), (1,), device=device)
            action = best[idx].item()

        return int(action)



    def evaluate(self, env: RectangleEnv, episodes=100, show_last_run = True):
        self.agent.device = next(self.agent.Q_net.parameters()).device
        # env = self.envs[env_name]
        eval_logs = []
        for ep in range(episodes):
            s = env.reset()
            # print(type(s))
            done, G, steps, iters = False, 0.0, 0, 0
            if ep == episodes - 1:
                env.render()
            while not done:
                # a = np.argmax(self.agent.Q[s])
                a = self.greedy_action(s)
                if env.is_stalled():
                    a = self.agent.fallback_action(s)
                render = True if (show_last_run and ep == episodes - 1) else False
                # print(render)
                s2, r, done = env.step(a, render=render)
                G += r
                s = s2
            eval_logs.append(
                {
                    "original_dist": env.inital_dist,
                    "ep": ep,
                    "reward": G,
                    "steps": env.steps,
                    "optimizer_iters": env.optimizer_iters,
                }
            )
        return eval_logs

    def plot_training_logs(
        self, col_name="reward", smooth_window=None, xlim=None, ylim=None
    ):
        df = pd.DataFrame(self.logs)

        plt.figure(figsize=(6, 4))

        # show training curve
        sns.lineplot(data=df, x="ep", y=col_name, alpha=0.7)
        if smooth_window:  # optional rolling mean
            df["smoothed"] = df[col_name].rolling(window=smooth_window).mean()
            sns.lineplot(
                data=df, x="ep", y="smoothed", label=f"smoothed ({smooth_window})"
            )
        plt.xlabel("Episode")
        plt.ylabel(col_name)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.title(f"Training {col_name} over Episodes")
        plt.show()
