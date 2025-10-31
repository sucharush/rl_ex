import matplotlib.pyplot as plt
import numpy as np
from src.environments.environment import RectangleEnv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

class RectangleRenderer:
    def __init__(self, env: RectangleEnv, figsize=(9,5)):
        self.env = env
        self.fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[5, 4], figure=self.fig)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.text_ax = self.fig.add_subplot(gs[0, 1])
        self.text_ax.axis("off")
    def render(self, return_rgb=False):
        self.ax.clear()
        self.env.rectangle.plot(self.ax, color="gray")
        self.env.rays.plot(ax=self.ax)
        self.env.segs.plot(show_points=True, ax=self.ax)
        self.ax.set_xlim(*self.env._xlim)
        self.ax.set_ylim(*self.env._ylim)
        self.ax.set_title(f"Step {self.env.steps}")

        # --- info text on the side ---
        info = self.env._get_info()
        self.text_ax.clear()
        self.text_ax.axis("off")

        # ---- convert all potentially array values to plain floats ----
        rect_center = np.array(info["rect_center"], dtype=float).ravel()
        ray_center  = np.array(info["ray_center"], dtype=float).ravel()
        rect_theta  = float(np.asarray(info["rect_theta_deg"]).item())
        ray_thetas  = np.array(info["ray_thetas_deg"], dtype=float).ravel()
        action = np.array(info["action"], dtype=float).ravel()

        # ---- build readable strings ----
        thetas_str = ", ".join(f"{float(t):.1f}" for t in ray_thetas)

        text = (
            f"Step: {int(info['steps'])}\n"
            f"Action: ({action[0]:.2f}, {action[1]:.2f}, {np.rad2deg(action[2]):.2f}°)\n"
            f"Rect Center: ({rect_center[0]:.2f}, {rect_center[1]:.2f})\n"
            f"Rect Theta: {rect_theta:.1f}°\n"
            f"Ray Center: ({ray_center[0]:.2f}, {ray_center[1]:.2f})\n"
            f"Ray Thetas: [{thetas_str}]°\n"
            f"Inside: {bool(info['ray_inside'])}"
        )

        # ---- draw text ----
        self.text_ax.text(
            0.0, 1.0, text,
            va="top", ha="left",
            fontsize=10,
            family="monospace",
        )
        self.fig.canvas.draw()


        if return_rgb:
            # Convert figure to RGB array
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            image = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(canvas.get_width_height()[::-1] + (4,))
            return image[..., :3]

