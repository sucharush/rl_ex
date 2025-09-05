import numpy as np
from scipy.optimize import least_squares
from src.environments.rectangle import Rectangle  # Assuming Rectangle is defined in rectangle.py


class StepOptimizer:
    def __init__(self, loss="soft_l1", bounds=None, k_softmin=None):
        self.loss = loss
        self.bounds = bounds  # e.g., (-0.5, 0.5)
        self.k_softmin = k_softmin  # if you choose soft-min instead of hinge

    def _validate_inputs(self, rect, points, direction):
        if not isinstance(rect, Rectangle):
            raise ValueError("rect must be an instance of Rectangle")
        if (
            not isinstance(points, np.ndarray)
            or points.ndim != 2
            or points.shape[1] != 2
        ):
            raise ValueError("points must be a 2D numpy array with shape (n, 2)")
        if direction not in ["vertical", "horizontal", "rotate"]:
            raise ValueError("direction must be 'vertical', 'horizontal' or 'rotate'")


    def _residuals(self, t, points:np.ndarray, direction:str, rect:Rectangle):
        shift = float(t[0])
        pts = points.copy()
        virtual_rect = rect.copy()

        dx, dy, theta = 0.0, 0.0, 0.0
        if direction in ("horizontal", "vertical"):
            horiz, vert = virtual_rect.get_local_directions()
            dvec = horiz if direction == "horizontal" else vert
            dx, dy = shift * dvec

        elif direction == "rotate":
            theta = shift

        virtual_rect.move(dx=dx, dy=dy, theta=theta)
        distance_new = virtual_rect.points_distance(pts, distance_only=True)

        return np.maximum(0.0, distance_new)

    def run(self, rect:Rectangle, points:np.ndarray, direction:str, max_nfev:int):
        self._validate_inputs(rect, points, direction)
        x0 = np.array([0.0])
        fun = lambda t: self._residuals(t, points, direction, rect=rect)
        res = least_squares(
            fun,
            x0,
            method="trf",
            loss=self.loss,
            max_nfev=max_nfev,
            bounds=self.bounds if self.bounds is not None else (-np.inf, np.inf),
        )
        delta = float(res.x[0])
        used = res.nfev
        return used, delta
