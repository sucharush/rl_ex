from collections import Counter
import numpy as np
from rectangle import Rectangle
from optimizer import StepOptimizer
import matplotlib.pyplot as plt
from IPython.display import Image, display
import imageio.v2 as imageio
import os


class RectangleEnv:
    def __init__(self, rect_params, optimizer_params, points, verbose=False):
        self.rect_params = rect_params
        self.optimizer_params = optimizer_params
        self.starting_points = points
        self.points = points
        self.max_steps = 30
        self.steps = 0
        self.latest_delta = 100  # large initial delta to ensure first move is always accepted
        self.latest_used_iters = None
        self.shift_steps = 0
        self.tau = optimizer_params.get("tau", 0.1)  # default tau
        self.close = 0.4
        self.manual_shift = 0.5  # manual step size for move actions
        self.manual_rotate = np.pi / 4  # manual rotation step size
        self.actions = [
            ("vertical", 2),
            ("vertical", 4),
            ("vertical", 6),
            ("horizontal", 2),
            ("horizontal", 4),
            ("horizontal", 6),
            ("move", ("up", 0.2)),
            ("move", ("down", 0.2)),
            ("move", ("left", 0.2)),
            ("move", ("right", 0.2)),
            ("move", ("up", 0.5)),
            ("move", ("down", 0.5)),
            ("move", ("left", 0.5)),
            ("move", ("right", 0.5)),
            ("rotate", 3),
            ("rotate", 6),
            ("rotate", 9),
            ("move", "rotate"),
        ]
        self.rectangle = Rectangle(**rect_params)
        self.optimizer = StepOptimizer(**optimizer_params)
        self._old_dist = self.rectangle.get_mean_dist(points=self.points)
        self.verbose_print = self.create_verbose(verbose)
        self.log = []
        self._xlim = None
        self._ylim = None
        self.state = None
        self.frames = []

    def create_verbose(self, verbose):
        def verbose_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        return verbose_print

    def reset(self):
        self.rectangle = Rectangle(**self.rect_params)
        self.points = self.starting_points.copy()
        self.steps = 0
        self.latest_used_iters = 0
        self._old_dist = self.rectangle.get_mean_dist(points=self.points)
        self.log = []
        self._xlim = None
        self._ylim = None
        self.frames = []
        return self.encode_state()

    def apply_action(self, action_id):
        # self.step_count += 1
        action = self.actions[action_id]

        direction, steps = action
        dx = dy = theta = 0.0
        if direction == "move":
            # print("direct_move")
            # pass
            if isinstance(steps, tuple):
                steps, delta = steps
            if steps == "up":
                dy = +delta
            elif steps == "down":
                dy = -delta
            elif steps == "right":
                dx = +delta
            elif steps == "left":
                dx = -delta
            elif steps == "rotate":
                # print("manual rotate")
                delta = self.manual_rotate
                theta = delta
            else:
                raise ValueError(f"Unknown move param: {steps}")
            logged_data = f"STEP {self.steps+1} >>>>>> Action: {action}, Delta: {delta}, Old Dist: {self._old_dist:.4f}"
            self.latest_used_iters = 0.5  # manual move always uses 1 iteration
        else:
            used, delta = self.optimizer.run(
                self.rectangle, self.points, direction, max_nfev=steps
            )
            logged_data = f"STEP {self.steps+1} >>>>>> Action: {action}, Used: {used}, Delta: {delta}, Old Dist: {self._old_dist:.4f}"
            # print(f"Rectangle info: {(self.rectangle.cx, self.rectangle.cy)} ")

            if direction == "vertical":
                dy = delta
                # self.rectangle.move(dy=-delta)
                # self.points[:, 1] += delta
            elif direction == "horizontal":
                dx = delta
                # self.rectangle.move(dx=-delta)
                # self.points[:, 0] += delta
            elif direction == "rotate":
                theta = delta
                used = used / 3
                # No change to points, rotation is around the rectangle's center
            self.latest_used_iters = used
        self.rectangle._latest_info = (
            (self.rectangle.cx, self.rectangle.cy),
            self.rectangle.theta,
        )
        self.rectangle.move(dx=dx, dy=dy, theta=theta)
        self.log.append(logged_data)
        self.latest_delta = delta

    def step(self, action):
        self.apply_action(action)
        state = self.encode_state()
        reward = self.compute_reward()
        self.steps += 1
        self.render()
        done = self.is_terminal()
        # print(done)
        return state, reward, done

    def show_gif(self, filename="animation.gif", fps=3):
        images = [imageio.imread(f) for f in self.frames]
        imageio.mimsave(filename, images, fps=fps)
        # display in notebook
        # display(Image(filename=filename))
        # # # cleanup
        # for f in self.frames:
        #     os.remove(f)

    def compute_reward(self, alpha=1.0, beta=0.5, stalled_pen=1):
        new_dist = self.rectangle.get_mean_dist(points=self.points)
        self.log.append(f"New Distance: {new_dist}")
        improv = self._old_dist - new_dist
        used = max(1, self.latest_used_iters)  # avoid division by zero
        reward = alpha * improv - beta * used
        # reward = alpha * (improv / used) - beta * used
        # reward = - self.latest_used_iters
        # Update memory so next step compares to this state's distance
        self._old_dist = new_dist
        return reward + 100 * self.is_all_touching()  # bonus for touching all sides

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

        return state

    def is_all_touching(self):
        return self.state[6] == len(self.points)  # num_touching == N
    
    def is_stalled(self):
        # Check if the rectangle has not moved significantly
        # or if the points have not changed position
        return self.latest_delta < 1e-6

    def is_terminal(self):
        # distances = np.array(self.rectangle.points_distance(points=self.points))
        # all_touching = np.all(distances < self.tau)  # match binning rule
        return self.is_all_touching() or (self.steps >= self.max_steps)

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

    def render(self, lock_square=False):
        fig, ax = plt.subplots()
        self.rectangle.plot(ax, color="gray")
        ax.scatter(self.points[:, 0], self.points[:, 1], color="red")
        ax.grid(True)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # print(f"step: {self.step_count}, limits: x=({x0:.2f}, {x1:.2f}), y=({y0:.2f}, {y1:.2f})")

        # Expand-only update of stored limits
        if self._xlim is None or self._ylim is None:
            self._xlim, self._ylim = (x0, x1), (y0, y1)
        else:
            self._xlim = (min(self._xlim[0], x0), max(self._xlim[1], x1))
            self._ylim = (min(self._ylim[0], y0), max(self._ylim[1], y1))

        # Optional: enforce square window by expanding the smaller span
        if lock_square:
            cx = 0.5 * (self._xlim[0] + self._xlim[1])
            cy = 0.5 * (self._ylim[0] + self._ylim[1])
            half = (
                max(self._xlim[1] - self._xlim[0], self._ylim[1] - self._ylim[0]) / 2.0
            )
            self._xlim = (cx - half, cx + half)
            self._ylim = (cy - half, cy + half)

        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_title(f"Step {self.steps}")
        # plt.show()
        fname = f"plots/frame_{self.steps}.png"
        fig.savefig(fname)
        plt.close(fig)  # don’t open windows
        self.frames.append(fname)
