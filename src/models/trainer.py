from environments.environment import RectangleEnv
import numpy as np
from collections import defaultdict
from models.agent_base import AgentBase
from typing import Dict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# class Trainer:
#     def __init__(
#         self,
#         envs: Dict[str, RectangleEnv] | list[RectangleEnv] | RectangleEnv,
#         agent: AgentBase,
#     ):
#         self.envs = envs
#         self.agent = agent
#         self.logs = []
#         # self.envs = dict((name, RectangleEnv(**env)) for name, env in envs.items())
#         if isinstance(envs, dict):
#             self.envs = envs
#         elif isinstance(envs, list):
#             self.envs = {f"env{i}": e for i, e in enumerate(envs)}
#         elif isinstance(envs, RectangleEnv):
#             self.envs = {"default": envs}

#     def train(self, episodes=1000, eps_schedule=None):
#         env_names = list(self.envs.keys())
#         for ep in range(episodes):
#             env = self.envs[env_names[ep % len(env_names)]]
#             s = env.reset()
#             a = self.agent.start_episode(s)
#             done, G, steps = False, 0.0, 0
#             while not done:
#                 s2, r, done = env.step(a)
#                 G += r
#                 steps += 1
#                 stalled = env.is_stalled()
#                 a = self.agent.step(s, a, r, s2, done, stalled)
#                 s = s2

#             # make sure any leftover transitions are updated
#             if hasattr(self.agent, "finish_episode"):
#                 self.agent.finish_episode(s)

#             self.logs.append(
#                 {"ep": ep, "reward": G, "steps": steps, "n_iters": env.optimizer_iters}
#             )
class Trainer:
    def __init__(
        self,
        envs: Dict[str, RectangleEnv] | list[RectangleEnv] | RectangleEnv | callable,
        agent: AgentBase,
        reuse_per_env: int = 1,
    ):
        """
        Either pass:
          - envs: dict/list/RectangleEnv (fixed pool), OR
          - env_generator: callable that returns a RectangleEnv
        """
        self.agent = agent
        self.logs = []
        self.reuse_per_env = reuse_per_env

        if callable(envs):   # env generator
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
                self.current_env = self.envs()   # call generator
            return self.current_env
        else:
            env_names = list(self.envs.keys())
            return self.envs[env_names[(ep // self.reuse_per_env) % len(env_names)]]

    def fit(self, episodes=1000, eps_schedule=None):
        # env_names = list(self.envs.keys()) if not self.is_generator else None
        for ep in range(episodes):
            env = self._get_env(ep)   # get fresh or reuse
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

            if hasattr(self.agent, "finish_episode"):
                self.agent.finish_episode(s)

            self.logs.append(
                {"ep": ep, "reward": G, "steps": steps, "n_iters": env.optimizer_iters}
            )


    def greedy_action(self, s):
        q = self.agent.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(np.random.choice(best))

    def evaluate(self, env: RectangleEnv, episodes=100):
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
                    "original_dist": env.inital_dist ,
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
