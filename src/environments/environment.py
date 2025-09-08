from collections import Counter
import numpy as np
from src.environments.rectangle import Rectangle
from src.environments.optimizer import StepOptimizer
import matplotlib.pyplot as plt
from IPython.display import Image, display
import imageio.v2 as imageio
import os


class RectangleEnv:
    def __init__(
        self,
        rect_params: dict,
        optimizer_params: dict,
        x_lim: tuple[float, float] = (-10, 10),
        y_lim: tuple[float, float] = (-10, 10),
        points: list[tuple[float, float]] = None,
        verbose: bool = False,
        max_episode_steps: int = 60,
        touching_dist: float = 0.2,
        close_dist: float = 0.5,
    ):
        self.rect_params = rect_params
        self.optimizer_params = optimizer_params
        self.rectangle = Rectangle(**rect_params)
        self.optimizer = StepOptimizer(**optimizer_params)
        self.starting_points = points
        self.points = points
        self.max_steps = max_episode_steps
        self.steps = 0
        self.latest_delta = (
            100  # large initial delta to ensure first move is always accepted
        )
        self.latest_used_iters = None
        self.optimizer_iters = 0
        self.shift_steps = 0
        self.tau = touching_dist  # default tau
        self.close = close_dist
        self.manual_shift = 0.5  # manual step size for move actions
        self.manual_rotate = np.pi / 4  # manual rotation step size
        self.actions = [
            # ("vertical", 2),
            ("vertical", 3),
            # ("vertical", 5),
            # ("horizontal", 2),
            ("horizontal", 3),
            # ("horizontal", 5),
            # ("move", ("up", 1)),
            # ("move", ("down", 1)),
            # ("move", ("left", 1)),
            # ("move", ("right", 1)),
            ("move", ("up", 0.5)),
            ("move", ("down", 0.5)),
            ("move", ("left", 0.5)),
            ("move", ("right", 0.5)),
            # ("rotate", 3),
            ("rotate", 4),
            # ("rotate", 8),
            # ("move", ("rotate", np.pi / 6)),
            ("move", ("rotate", np.pi / 6)),
            ("move", ("rotate", -np.pi / 6)),
        ]
        self.rectangle.points_distance(self.points)
        self._old_dist = self.rectangle.get_mean_dist(points=self.points)
        self.verbose_print = self.create_verbose(verbose)
        self.log = []
        self._xlim = x_lim
        self._ylim = y_lim
        self.state = None
        self.inital_dist = self._old_dist
        self.prev_location = []
        self.frames = []

    def create_verbose(self, verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        return verbose_print

    def reset(self):
        self.rectangle = Rectangle(**self.rect_params)
        self.points = self.starting_points.copy()
        self.rectangle.points_distance(self.points)
        self.steps = 0
        self.latest_used_iters = 0
        self.optimizer_iters = 0
        self._old_dist = self.rectangle.get_mean_dist(points=self.points)
        self.inital_dist = self._old_dist
        self.log = []
        self.frames = []
        self.prev_location = []
        return self.encode_state()  # state is updated inside encode_state()

    def apply_action(self, action_id):
        action = self.actions[action_id]
        direction, info = action
        dx = dy = theta = 0.0
        horiz, vert = self.rectangle.get_local_directions()
        dvec = None
        if direction == "move":
            if isinstance(info, tuple):
                steps, delta = info
            if steps == "up":
                dx, dy = delta * vert
            elif steps == "down":
                dx, dy = -delta * vert
            elif steps == "right":
                dx, dy = delta * horiz
            elif steps == "left":
                dx, dy = -delta * horiz
            elif steps == "rotate":
                theta = delta
            else:
                raise ValueError(f"Unknown move param: {steps}")
            logged_data = f"STEP {self.steps+1} >>>>>> Action: {action}, Delta: {delta}, Old Dist: {self._old_dist:.4f}"
            self.latest_used_iters = 0.5  # manual move always uses 0.5 iteration
        else:
            used, delta = self.optimizer.run(
                self.rectangle, self.points, direction, max_nfev=info
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
            self.latest_used_iters = used


        self.rectangle.move(dx=dx, dy=dy, theta=theta)
        self.rectangle.points_distance(self.points) # update the distance lists right after the move
        self.log.append(logged_data)
        self.latest_delta = delta

    def step(self, action, render=False):
        # store history (latest 3 only)
        self.prev_location.append(self.rectangle.get_info())
        if len(self.prev_location) > 3:
            self.prev_location.pop(0)
        self.apply_action(action)
        state = self.encode_state()
        reward = self.compute_reward()
        self.steps += 1
        if render:
            self.render()
        done = self.is_terminal()
        # print(done)
        return state, reward, done

    def show_gif(self, filename="animation.gif", fps=3, pause_frames=5):
        images = [imageio.imread(f) for f in self.frames]

        # duplicate last frame to create pause
        images.extend([images[-1]] * pause_frames)

        imageio.mimsave(
            filename,
            images,
            fps=fps,
            loop=0  # loop forever
        )


    def compute_reward(self, alpha=1.0, beta=0.2, stalled_pen=1):
        new_dist = self.rectangle.get_mean_dist(points=self.points)
        self.log.append(f"New Distance: {new_dist}")
        improv = self._old_dist - new_dist
        used = max(1, self.latest_used_iters)  # avoid division by zero
        reward = alpha * improv - beta * used
        self._old_dist = new_dist
        return (
            reward + 10 * self.is_all_touching() - 10 * self.is_out_of_bounds()
        )  # bonus for touching all sides

    def encode_state(self):
        # Count by closest side
        side_counts = Counter(
            self.rectangle.get_side_list(points=self.points)
        )  # e.g., ["top","left",...]
        num_top = side_counts.get("top", 0)
        num_bottom = side_counts.get("bottom", 0)
        num_right = side_counts.get("right", 0)
        num_left = side_counts.get("left", 0)

        # Bin distances: 0=touching (<tau), 1=close ([tau, close)), 2=far (>=close)
        d = self.rectangle.get_distance_list(points=self.points)
        binned = [0 if x < self.tau else (1 if x < self.close else 2) for x in d]

        dist_counts = Counter(binned)
        num_touching = dist_counts.get(0, 0)
        num_close = dist_counts.get(1, 0)
        num_far = dist_counts.get(2, 0)

        # Optional safety checks (can disable in prod)
        N = len(d)
        assert (
            num_top + num_bottom + num_right + num_left
        ) == N, "side counts mismatch"
        assert (num_touching + num_close + num_far) == N, "distance counts mismatch"

        state = (
            num_top,
            num_bottom,
            num_right,
            num_left,
            num_far,
            num_close,
            num_touching,
        )
        self.state = state
        self.log.append(f"State: {state}")

        return state

    def is_out_of_bounds(self):
        x_min, x_max = self._xlim[0], self._xlim[1]
        y_min, y_max = self._ylim[0], self._ylim[1]
        x, y = self.rectangle.get_corners().T
        return x.min() < x_min or x.max() > x_max or y.min() < y_min or y.max() > y_max

    def is_all_touching(self):
        return self.state[6] == len(self.points)  # num_touching == N

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
        return False

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        self.rectangle.plot(ax, color="gray")
        ax.scatter(self.points[:, 0], self.points[:, 1], color="red")
        ax.set_title(f"Step {self.steps}")
        plt.show()

    def clear_view_lock(self):
        """Call this if you want to recompute the view on the next render()."""
        self._xlim = self._ylim = None
        self._xticks = self._yticks = None

    def render(self, show=False):
        fig, ax = plt.subplots()
        self.rectangle.plot(ax, color="gray")
        ax.scatter(self.points[:, 0], self.points[:, 1], color="blue")
        ax.grid(True)

        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_title(f"Step {self.steps}")

        if show:
            plt.show()
        else:
            fname = f"plots/frame_{self.steps}.png"
            fig.savefig(fname)
            plt.close(fig)
            self.frames.append(fname)
