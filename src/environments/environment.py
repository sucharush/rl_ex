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
        segs: Segmentation,
        x_lim: tuple[float, float] = (-10, 10),
        y_lim: tuple[float, float] = (-10, 10),
        verbose: bool = False,
        max_episode_steps: int = 60,
        touching_dist: float = 0.3,
        close_dist: float = 0.8,
    ):
        self.rect_params = rect_params
        self.optimizer_params = optimizer_params
        self.rays_params = rays_params
        self.segs = segs
        self.rectangle = Rectangle(**rect_params)
        self.optimizer = StepOptimizer(**optimizer_params)
        self.rays = RayBatch.from_rectangle(
            rect=self.rectangle, segs=self.segs, **rays_params
        )
        self.starting_points = self.segs.points
        self.points = self.segs.points
        self.max_steps = max_episode_steps
        self.steps = 0
        self.latest_delta = (
            60  # large initial delta to ensure first move is always accepted
        )
        self.latest_used_iters = None
        self.latest_action = None
        self.optimizer_iters = 0
        self.shift_steps = 0
        self.tau = touching_dist  # default tau
        self.close = close_dist
        self.manual_shift = 0.5  # manual step size for move actions
        self.manual_rotate = np.pi / 4  # manual rotation step size
        self.actions = [
            ("vertical", 4),
            ("horizontal", 4),
            ("move", ("up", 1)),
            ("move", ("down", 1)),
            ("move", ("left", 1)),
            ("move", ("right", 1)),
            # ("move", ("up", 0.5)),
            # ("move", ("down", 0.5)),
            # ("move", ("left", 0.5)),
            # ("move", ("right", 0.5)),
            ("rotate", 4),
            ("move", ("rotate", np.pi / 6)),
            ("move", ("rotate", -np.pi / 6)),
            # ("all", 20),  # 10 iterations of all-direction optimization
        ]
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

    def create_verbose(self, verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        return verbose_print

    def reset(self):
        # ========================
        # TODO: CHANGE ACCORDINGLY
        # ========================
        self.rectangle = Rectangle(**self.rect_params)
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
        return self.encode_state()  # state is updated inside encode_state()

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

    def apply_action(self, action_id):
        action = self.actions[action_id]
        direction, info = action
        dx = dy = theta = 0.0
        horiz, vert = self.rectangle.get_local_directions()
        used, dvec = 0, None
        if direction == "move":
            if isinstance(info, tuple):
                steps, delta = info
            if steps == "up":
                dx, dy = delta * vert
                used = 1
            elif steps == "down":
                dx, dy = -delta * vert
                used = 1
            elif steps == "right":
                dx, dy = delta * horiz
                used = 1
            elif steps == "left":
                dx, dy = -delta * horiz
                used = 1
            elif steps == "rotate":
                theta = delta
                used = 2  # rotation is less risky
            else:
                raise ValueError(f"Unknown move param: {steps}")
            logged_data = f"STEP {self.steps+1} >>>>>> Action: {action}, Delta: {delta}, Old Dist: {self._old_dist:.4f}"
            self.latest_used_iters = used  # manual move always uses 1 iteration
        else:
            # print("hello from env.apply_action")
            used, delta = self.optimizer.run(
                rect=self.rectangle,
                points=self.segs.points,
                direction=direction,
                max_nfev=info,
            )

            logged_data = f"STEP {self.steps+1} >>>>>> Action: {action}, Used: {used}, Delta: {delta}, Old Dist: {self._old_dist:.4f}"
            self.optimizer_iters += used

            if direction == "vertical":
                dvec = vert
                dx, dy = delta * dvec

            elif direction == "horizontal":
                dvec = horiz
                dx, dy = delta * dvec

            elif direction == "rotate":
                theta = delta
                used = used
                # No change to points, rotation is around the rectangle's center
            elif direction == "all":
                dx, dy, theta = delta
                used = used
            self.latest_used_iters = used

        self.rectangle.move(dx=dx, dy=dy, theta=theta)
        # self.rectangle.points_signed_segment_distance(
        #     self.points
        # )  # update the distance lists right after the move
        self.log.append(logged_data)
        self.latest_delta = delta

    def step(self, action, render=False):
        # store history (latest 3 only)
        self.prev_location.append(self.rectangle.get_info())
        if len(self.prev_location) > 4:
            self.prev_location.pop(0)
        self.apply_action(action)
        self.latest_action = action
        self.update_rays()
        self.update_dist()
        state = self.encode_state()
        self.debug_temp()
        reward = self.compute_reward()
        self.steps += 1
        if render:
            self.render()
        done = self.is_terminal()
        return state, reward, done

    def show_gif(self, filename="animation.gif", fps=3, pause_frames=5):
        print(f"Saving GIF to {filename} ...")
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
    ):
        # distances
        D_prev = self._old_dist
        D_cur = self._cur_dist
        D0 = max(self.inital_dist, 1e-8)  # store at episode start

        # normalized terms
        dist_term = -(D_cur / D0)  # in ~[-?, 0]
        improv_raw = (D_prev - D_cur) / D0
        improv_term = float(np.clip(improv_raw, -clip_improv, clip_improv))

        # action-cost (lighter for direct moves, heavier for optimizer)
        used = max(1, self.latest_used_iters)
        if self.is_rotation(self.latest_action):
            # penalty_scale = 1 + rot_factor * max(0, (rot_improv_tol - improv_raw) / rot_improv_tol)
            penalty_scale = 1.5 - 0.3 * self.is_center_inside()
            used *= penalty_scale
        step_cost = beta * used
        # bonuses
        touch = float(self.is_all_touching())
        oob = float(self.is_out_of_bounds())

        r = (
            dist_term
            + alpha * improv_term
            - step_cost
            + bonus_touch * touch
            - penalty_oob * oob
        )
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
            and np.mean(np.abs(dist_ls) <= self.tau) >= 0.7
            and np.mean(np.abs(dist_ls) <= self.close) >= 0.9
        ):
            return True
        return False

    def is_center_inside(self):
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
            return True
        if self.is_all_touching():
            self.log.append("Terminal: All points touching")
            return True
        if self.steps >= self.max_steps:
            self.log.append("Terminal: Max steps reached")
            return True
        # if self.is_center_inside():
        #     self.log.append("Terminal: Center inside")
        #     return True
        return False

    def clear_view_lock(self):
        """Call this if you want to recompute the view on the next render()."""
        self._xlim = self._ylim = None
        self._xticks = self._yticks = None

    def render(self, show=False):
        fig, ax = plt.subplots()
        self.rectangle.plot(ax, color="gray")
        self.rays.plot()
        self.segs.plot(show_points=True)
        # ax.scatter(self.points[:, 0], self.points[:, 1], color="blue")
        ax.grid(True)

        # ax.set_xlim(*self._xlim)
        # ax.set_ylim(*self._ylim)
        ax.set_title(f"Step {self.steps}")
        # print(self._xlim, self._ylim)
        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)

        if show:
            plt.show()
        else:
            fname = f"plots/frame_{self.steps}.png"
            fig.savefig(fname)
            plt.close(fig)
            self.frames.append(fname)
