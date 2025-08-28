from environments.environment import RectangleEnv
import numpy as np
from collections import defaultdict
from models.agent_base import AgentBase
from typing import Dict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        envs: Dict[str, RectangleEnv] | list[RectangleEnv] | RectangleEnv,
        agent: AgentBase,
    ):
        self.envs = envs
        self.agent = agent
        self.logs = []
        # self.envs = dict((name, RectangleEnv(**env)) for name, env in envs.items())
        if isinstance(envs, dict):
            self.envs = envs
        elif isinstance(envs, list):
            self.envs = {f"env{i}": e for i, e in enumerate(envs)}
        elif isinstance(envs, RectangleEnv):
            self.envs = {"default": envs}

    def train(self, episodes=1000, eps_schedule=None):
        env_names = list(self.envs.keys())
        for ep in range(episodes):
            env = self.envs[env_names[ep % len(env_names)]]
            s = env.reset()
            a = self.agent.start_episode(s)
            done, G, steps = False, 0.0, 0
            while not done:
                s2, r, done = env.step(a)
                G += r
                steps += 1
                stalled = env.is_stalled()
                a = self.agent.step(s, a, r, s2, done, stalled)
                s = s2

            # 🔑 make sure any leftover transitions are updated
            if hasattr(self.agent, "finish_episode"):
                self.agent.finish_episode(s)

            self.logs.append(
                {"ep": ep, "reward": G, "steps": steps, "n_iters": env.optimizer_iters}
            )

    def greedy_action(self, s):
        q = self.agent.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(np.random.choice(best))

    def evaluation(self, env: RectangleEnv, episodes=100):
        # env = self.envs[env_name]
        eval_logs = []
        for ep in range(episodes):
            s = env.reset()
            done, G, steps, iters = False, 0.0, 0, 0
            env.render()
            while not done:
                # a = np.argmax(self.agent.Q[s])
                a = self.greedy_action(s)
                if env.is_stalled():
                    a = self.agent.fallback_action()
                s2, r, done = env.step(a, render=True)
                G += r
                # steps += 1
                # iters += int(env.latest_used_iters or 0)
                s = s2
            eval_logs.append(
                {
                    "ep": ep,
                    "reward": G,
                    "steps": env.steps,
                    "optimizer_iters": env.optimizer_iters,
                }
            )
        return eval_logs

    def plot_training_logs(
        self, col_name="return", smooth_window=None, xlim=None, ylim=None
    ):
        # values = [log[col_name] for log in self.logs]
        # df = pd.DataFrame({"episode": range(len(values)), col_name: values})
        df = pd.DataFrame(self.logs)

        plt.figure(figsize=(8, 4))

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
