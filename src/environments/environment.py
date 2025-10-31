from collections import Counter
import numpy as np
from src.environments.rectangle import Rectangle
from src.environments.optimizer import StepOptimizer
from src.environments.points import PointSet
from src.environments.segments import Segmentation
from src.environments.rays import RayBatch
import matplotlib.pyplot as plt
from IPython.display import Image, display
import imageio.v2 as imageio
import os
from typing import Optional


class RectangleEnv:
    def __init__(
        self,
        rect_params: dict,
        optimizer_params: dict,
        rays_params: dict,
        segs_params: dict,
        # segs: Segmentation,
        x_lim: tuple[float, float] = (-10, 10),
        y_lim: tuple[float, float] = (-10, 10),
        verbose: bool = False,
        max_episode_steps: int = 60,
        touching_dist: float = 0.3,
        close_dist: float = 0.5,
        far_dist: float = 2,
        action_scale=(1.2, 1.2, np.pi / 4),
        hard_reset: bool = False,
    ):
        self.rect_params = rect_params
        self.optimizer_params = optimizer_params
        self.rays_params = rays_params
        self.segs_params = segs_params
        # self.segs = segs
        self.rectangle = Rectangle(**rect_params)
        self.optimizer = StepOptimizer(**optimizer_params)
        # self._seg_generator = self._build_seg_generator()
        self._rng = np.random.default_rng(segs_params.get("seed", None))
        self._build_seg_generator()
        segs = self._seg_generator()
        self.segs_copy = segs
        self.segs = segs
        self.rays = RayBatch.from_rectangle(
            rect=self.rectangle, segs=self.segs, **rays_params
        )
        self.starting_points = self.segs.points
        self.points = self.segs.points
        self.max_steps = max_episode_steps
        self.steps = 0
        self.latest_used_iters = None
        self.optimizer_iters = 0
        self.shift_steps = 0
        self.tau = touching_dist  # default tau
        self.close = close_dist
        self.far = far_dist
        self.manual_shift = 0.5  # manual step size for move actions
        self.manual_rotate = np.pi / 4  # manual rotation step size
        self.latest_action = np.array([0, 0, 0])
        self.action_scale = np.array(action_scale)
        # self.rectangle.points_signed_segment_distance(self.points)
        # initialize distances
        self.dist_list = self.rectangle.points_distance(self.points)
        self._cur_dist = np.mean(self.dist_list)
        self._old_dist = self._cur_dist
        self.verbose_print = self.create_verbose(verbose)
        self.log = []
        self._xlim = x_lim
        self._ylim = y_lim
        self.state = self.encode_state()
        self.inital_dist = self._old_dist
        self.prev_location = []
        self.frames = []
        self.hard_reset = hard_reset
        self.inside_flag = 0

    def create_verbose(self, verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        return verbose_print
    def set_rng(self, rng: np.random.Generator, seed: int | None = None):
        """Synchronize RNG with outer Gym environment."""
        self._rng = rng
        self._rng_seed = seed

    def reset(self, seed: int | None = None):
        """Reset environment with optional seed (Gym-compatible)."""
        if seed is not None:
            # If Gym passes a fixed seed, reinitialize RNG deterministically
            self._rng = np.random.default_rng(seed)
        elif not hasattr(self, "_rng") or self._rng is None:
            # Fallback: create a persistent RNG if missing
            self._rng = np.random.default_rng()

        self.rectangle = Rectangle(**self.rect_params)
        # Use _rng for sampling instead of np.random.default_rng()
        seg_seed = self._rng.integers(1e9)
        # print(seg_seed)
        self.segs = self.segs_copy if self.hard_reset else self._seg_generator(seg_seed)
        self.rays = RayBatch.from_rectangle(
            rect=self.rectangle, segs=self.segs, **self.rays_params
        )
        self.points = self.segs.points
        self.steps = 0
        self.latest_used_iters = 0
        self.optimizer_iters = 0
        self.dist_list = self.rectangle.points_distance(self.points)
        self._cur_dist = np.mean(self.dist_list)
        self._old_dist = self._cur_dist
        self.inital_dist = self._cur_dist
        self.log = []
        self.frames = []
        self.prev_location = []
        self.inside_flag = 0
        self.latest_action = np.array([0, 0, 0])
        return self.encode_state()  # state is updated inside encode_state()

    def _get_info(self, print_info: bool = False) -> dict:
        """
        Collect diagnostic information about the current environment state.

        Parameters
        ----------
        print : bool
            If True, print the information to stdout.

        Returns
        -------
        info : dict
            A dictionary containing geometric and environment statistics.
        """
        rect_center = self.rectangle.get_center()       # (cx, cy)
        rect_theta = np.rad2deg(self.rectangle.theta)   # in degrees
        ray_center = self.rays.origin                   # (x0, y0)
        ray_inside = self.rays.center_inside             # bool
        ray_thetas = np.rad2deg(self.rays.thetas)          # per-ray directions in degrees
        # action = np.array([self.latest_action[0], self.latest_action[1], np.degrees(self.action_scale[2])*self.latest_action[3]])

        info = {
            "rect_center": rect_center,
            "rect_theta_deg": rect_theta,
            "ray_center": ray_center,
            "ray_inside": bool(ray_inside),
            "ray_thetas_deg": ray_thetas.tolist(),
            "steps": self.steps,
            "action": self.latest_action,
            "done": getattr(self, "_done", False),
        }

        if print_info:
            print(f"[Step {self.steps}] Rect center: {rect_center}, θ={rect_theta:.2f}°")
            print(f"  Ray origin: {ray_center}, inside={ray_inside}")

        return info

    def _build_seg_generator(self):
        offset_range, rotation_range = (
            self.segs_params["offset_range"],
            self.segs_params["rotation_range"],
        )
        num_points, jitter_n, jitter_t = (
            self.segs_params["num_points"],
            self.segs_params["jitter_n"],
            self.segs_params["jitter_t"],
        )

        def _generator(seed=None):
            # print(seed)
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            rect = self.rectangle.copy()
            dx = rng.uniform(*offset_range[0])
            dy = rng.uniform(*offset_range[1])
            rot = rng.uniform(*rotation_range)
            rect.move(dx=dx, dy=dy, theta=rot)
            nodes = rect.sample(num_points=num_points, jitter_n=jitter_n, jitter_t=jitter_t, rng=rng)
            # print(nodes)
            return Segmentation.from_nodes(nodes, num_samples=20)

        self._seg_generator = _generator


    def update_dist(self):
        self.dist_list = self.rectangle.points_distance(self.points)
        self._old_dist = self._cur_dist
        self._cur_dist = np.mean(self.dist_list)
        # print(self._old_dist, self._cur_dist)

    def update_rays(self):
        theta0, theta1, is_inside = self.rectangle.angle_span_from_points(
            segs=self.segs
        )
        self.rays.update_position(
            origin=self.rectangle.get_center(),
            theta0=theta0,
            theta1=theta1,
            is_inside=is_inside,
        )

    def apply_action(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        scaled = action * self.action_scale
        self.latest_action = scaled
        dx_local, dy_local, theta = scaled
        dx, dy = self.rectangle.get_direction() @ np.array([dx_local, dy_local])
        # dx, dy = dx_local, dy_local
        self.rectangle.move(dx=dx, dy=dy, theta=theta)
        self.latest_used_iters = 1

        # self.rectangle.points_signed_segment_distance(
        #     self.points
        # )  # update the distance lists right after the move
        logged_data = f"STEP {self.steps+1} >>>>>> dx: {dx}, dy: {dy}, dtheta: {theta}, Old Dist: {self._old_dist:.4f}"
        self.log.append(logged_data)
        # print(logged_data)

    def step(self, action: np.ndarray, render=False):
        # store history (latest 3 only)
        self.prev_location.append(self.rectangle.get_info())
        if len(self.prev_location) > 4:
            self.prev_location.pop(0)
        self.apply_action(action)
        self.update_rays()
        self.update_dist()
        state = self.encode_state()
        # self._get_info(print_info=True)
        # self.debug_temp()
        reward = self.compute_reward()
        self.steps += 1
        if render:
            self.render()
        done = self.is_terminal()
        return state, reward, done

    def show_gif(self, filename="animation.gif", fps=3, pause_frames=5):
        images = [imageio.imread(f) for f in self.frames]
        # duplicate last frame to create pause
        images.extend([images[-1]] * pause_frames)
        imageio.mimsave(filename, images, fps=fps, loop=0)  # loop forever

    def is_rotation(self, action: int):
        move, info = self.actions[action]
        if move == "rotate":
            return True
        elif move == "move" and info[0] == "rotate":
            return True
        return False

    def compute_reward(
        self,
        alpha=0.3,  # improvement weight
        beta=0.4,  # step penalty
        bonus_touch=5.0,
        penalty_oob=3.0,
        clip_improv=1.0,
        rot_improv_tol=0.3,
        rot_factor=2,
    ):
        # distances
        D_prev = self._old_dist
        D_cur = self._cur_dist
        D0 = max(self.inital_dist, 1e-1)  # store at episode start
        # print(D_cur, D0)

        # normalized terms
        dist_term = -(D_cur / D0)  # in ~[-?, 0]
        improv_raw = (D_prev - D_cur) / D0
        improv_term = float(np.clip(improv_raw, -clip_improv, clip_improv))

        # action-cost (lighter for direct moves, heavier for optimizer)
        used = max(1, self.latest_used_iters)
        step_cost = beta * used

        # bonuses
        bonus_inside = beta / 2
        touch = float(self.is_all_touching())
        oob = float(self.is_out_of_bounds())
        inside = float(self.is_center_inside())

        r = (
            # dist_term
            + alpha * improv_term
            - step_cost
            + bonus_touch * touch
            + bonus_inside * inside
            - penalty_oob * oob
        )
        # print(r)
        self._old_dist = D_cur
        self.log.append(f"Reward: {r:.3f}")
        return r

    def encode_state(self):
        # print(self.segs.segments)
        gaps = self.rays.compute_gap(rect=self.rectangle, segments=self.segs.segments)
        thatas = self.rays.thetas
        first_dir = self.rays.dirs[0]
        last_dir = self.rays.dirs[-1]
        rect_dir = self.rectangle.get_direction()
        # print("[RectangleEnv.encode_state] thata", self.rectangle.theta)
        local_dir1 = [first_dir @ rect_dir[0], first_dir @ rect_dir[1]]
        local_dir2 = [last_dir @ rect_dir[0], last_dir @ rect_dir[1]]
        # print("local_dir", local_dir)
        local_dir = self.rays.dirs @ rect_dir.T  # shape (n_rays, 2)
        span = max(thatas) - min(thatas)
        state = np.r_[gaps[0], gaps[1], local_dir.flatten()]
        self.state = state
        if np.isnan(state).any():
            print("[RectangleEnv.encode_state]: NaN in state")
            self.render(show = True)
        # self._get_info(print_info=True)
        return state

    def debug_temp(self):
        if np.any(self.state == np.inf):
            self.render(show=True)

    def is_out_of_bounds(self):
        x_min, x_max = self._xlim[0], self._xlim[1]
        y_min, y_max = self._ylim[0], self._ylim[1]
        x, y = self.rectangle.get_corners().T
        return x.min() < x_min or x.max() > x_max or y.min() < y_min or y.max() > y_max

    def is_all_touching(self):
        len = 2 * self.rays.n_rays
        dist_ls = self.state[:len]
        if (
            self.is_center_inside()
            and np.mean(np.abs(dist_ls) <= self.tau) >= 0.5
            and np.mean(np.abs(dist_ls) <= self.close) >= 0.8
            and np.mean(np.abs(dist_ls) <= self.far) >= 0.9
        ):
            return True
        return False

    def is_center_inside(self):
        # print("[RectangleEnv.is_center_inside]:", self.rays.center_inside)
        return self.rays.center_inside

    def is_stalled(self):
        """Check if current rectangle pose matches one of the last two stored poses."""
        if len(self.prev_location) < 2:
            return False

        current = self.rectangle.get_info()
        return any(np.allclose(current, prev, atol=1e-3) for prev in self.prev_location)

    def is_terminal(self):
        if self.is_out_of_bounds():
            self.log.append("Terminal: Out of bounds")
            # print("Terminal: Out of bounds")
            return True
        if self.is_all_touching():
            self.log.append("Terminal: All points touching")
            # print("Terminal: All points touching")
            return True
        if self.steps >= self.max_steps:
            self.log.append("Terminal: Max steps reached")
            # print("Terminal: Max steps reached")
            return True
        # if self.is_center_inside():
        #     self.log.append("Terminal: Center inside")
        #     return True
        return False

    def clear_view_lock(self):
        """Call this if you want to recompute the view on the next render()."""
        self._xlim = self._ylim = None
        self._xticks = self._yticks = None

    # def render(self, show=False):
    #     fig, ax = plt.subplots()
    #     self.rectangle.plot(ax, color="gray")
    #     self.rays.plot()
    #     self.segs.plot(show_points=True)
    #     # ax.scatter(self.points[:, 0], self.points[:, 1], color="blue")
    #     ax.grid(True)

    #     # ax.set_xlim(*self._xlim)
    #     # ax.set_ylim(*self._ylim)
    #     ax.set_title(f"Step {self.steps}")
    #     # print(self._xlim, self._ylim)
    #     ax.set_xlim(self._xlim)
    #     ax.set_ylim(self._ylim)

    #     if show:
    #         plt.show()
    #     else:
    #         fname = f"plots/frame_{self.steps}.png"
    #         fig.savefig(fname)
    #         plt.close(fig)
    #         self.frames.append(fname)
