import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.environments.environment import RectangleEnv   # your original environment
from src.environments.renderer import RectangleRenderer
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")   # or "QtAgg" depending on your system
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# plt.ion()

class GymRectangleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, rect_params, optimizer_params, rays_params, segs_params,
                #  seed = 42,
                 x_lim=(-10, 10), y_lim=(-10, 10),
                 max_episode_steps=60, verbose=False, render_mode = None):

        super().__init__()

        # underlying environment
        self._env = RectangleEnv(
            rect_params=rect_params,
            optimizer_params=optimizer_params,
            rays_params=rays_params,
            segs_params=segs_params,
            x_lim=x_lim,
            y_lim=y_lim,
            verbose=verbose,
            max_episode_steps=max_episode_steps,
        )

        # Action space: discrete integer IDs
        # self.action_space = spaces.Discrete(len(self._env.actions))
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: continuous vector of distances + directions
        # Here the state is a flat NumPy array from encode_state()
        state_dim = len(self._env.encode_state())
        high = np.inf * np.ones(state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.renderer = RectangleRenderer(self._env)
        self.render_mode = render_mode
        self.env_time = 0
        if self.render_mode == "human":
            matplotlib.use("TkAgg")
        elif self.render_mode == "rgb_array":
            matplotlib.use("Agg")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # state = self._env.reset()
        self._env.set_rng(self.np_random, self.np_random_seed)
        state = self._env.reset(seed=seed)
        info = self._env._get_info()
        return np.array(state, dtype=np.float32), info   # Gymnasium requires (obs, info)

    def step(self, action:np.ndarray):
        cur = time.time()
        state, reward, done = self._env.step(action)
        cost = time.time() - cur
        self.env_time += cost
        # print(done, state)
        truncated = False
        info = {}
        return np.array(state, dtype=np.float32), float(reward), done, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            return self.renderer.render(return_rgb=True)

    def render(self, sec = 0.3):
        if self.renderer is None:
            self.renderer = RectangleRenderer(self._env)

        if self.render_mode == "human":
            # print("Calling renderer (human)")
            self.renderer.render(return_rgb=False)
            plt.pause(sec)  # make sure it updates

        elif self.render_mode == "rgb_array":
            # print("Calling renderer (rgb_array)")
            return self.renderer.render(return_rgb=True)
        else:
            # if render_mode=None, do nothing
            return None

    def close(self):
        pass