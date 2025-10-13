import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from src.utils.geometry import rays_segments_span
from src.environments.rectangle import Rectangle
from src.environments.segments import Segmentation


class RayBatch:
    def __init__(self, n_rays: int = 5, default_dist=None):
        self.origin = None  # (2,)
        self.dirs = None  # (M,2)
        self.thetas = None  # (M,)
        self.default_dist = default_dist
        self.center_inside = None
        self.n_rays = n_rays
        # self.pad = pad

    @classmethod
    def from_rectangle(
        cls, rect: Rectangle, segs: Segmentation, n_rays=5
    ):
        """
        Factory: build rays aligned to a rectangle and a pointset.
        """
        rays = cls(n_rays=n_rays)

        # 1) span from points
        theta0, theta1, center_inside = rect.angle_span_from_points(segs)
        # rays.center_inside = center_inside

        # print(theta0, theta1)
        rays.update_position(rect.get_center(), theta0, theta1, center_inside)

        # 2) default span + distances
        theta_df0, theta_df1 = rect.get_default_range()
        dirs, _ = rays.compute_dirs(theta0=theta_df0, theta1=theta_df1)
        rays.default_dist = rect.internal_dist(dirs)
        return rays

    def compute_dirs(self, theta0, theta1):
        # pad = self.pad
        gap = np.mod(np.abs(theta0 - theta1), 2*np.pi)
        gap = min(gap, 2*np.pi-gap)
        pad = gap/(self.n_rays*2)
        # print("original", theta0, theta1)
        # print("pad", pad)
        # if theta0 > theta1 and abs(theta0 - theta1) != np.pi:
        #     theta0 -= 2 * np.pi
        #     thetas = np.linspace(theta0 + pad, theta1 - pad, self.n_rays, endpoint=True)
        # else:
        #     thetas = np.linspace(theta0 + pad, theta1 - pad, self.n_rays, endpoint=True)
        # print("processed",thetas.min(),thetas.max())
        # print("original", theta0, theta1)
        dtheta = (theta1 - theta0) % (2*np.pi)
        if dtheta > np.pi:
            dtheta -= 2*np.pi   # choose the shorter way around
        # thetas = np.linspace(theta0 + pad, theta0 + dtheta - pad, self.n_rays, endpoint=True)
        # print(theta0, dtheta)
        thetas = np.linspace(theta0, theta0+dtheta, self.n_rays + 2, endpoint=True)[1:-1]
        # print("processed",thetas.min(),thetas.max())
        dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
        return dirs, thetas

    @staticmethod
    def one_side_gap(raw, internal):
        return np.where(raw >= 0, raw - internal, -(raw + internal))

    def compute_gap(self, rect: Rectangle, segments: np.ndarray):
        raw_min, raw_max = self.compute_segments_dist(segments=segments)
        # print("[RayBatch.compute_gap] raw_left", raw_left)
        # print("[RayBatch.compute_gap] raw_right", raw_right)
        if self.center_inside:
            internal = self.default_dist
        else:
            internal = rect.internal_dist(self.dirs)
        # print(raw_min, raw_max, internal)
        # print(raw_left, raw_right, internal)
        return self.one_side_gap(raw_min, internal), self.one_side_gap(raw_max, internal)

    def update_position(self, origin, theta0, theta1, is_inside):
        """
        Build rays evenly spaced between [theta0, theta1].
        rect_dist_fn: function(origin, dirs) -> (M,) distances to rectangle boundary
        """
        self.origin = origin
        self.center_inside = is_inside
        self.dirs, self.thetas = self.compute_dirs(theta0, theta1)

    def compute_segments_dist(self, segments: np.ndarray):
        # o = self.origin
        p1s, p2s = segments[:, 0], segments[:, 1]
        # print("hello from rays")
        # print(self.origin, self.dirs, p1s.shape, p2s.shape)
        seg_dists = rays_segments_span(o=self.origin, dirs=self.dirs, p1s=p1s, p2s=p2s)
        # print(seg_dists)
        return seg_dists

    def plot(self, ax=None, length=None, **kwargs):
        """
        Plot rays extending across the plane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on (must already exist).
        length : float
            Half-length in both directions along the ray.
        kwargs : dict
            Extra arguments forwarded to ax.plot (e.g. color, linewidth).
        """
        if ax is None:
            ax = plt.gca()
        ox, oy = self.origin

        if length is None:
            # Use current axis limits to determine span
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            length = 1.5 * max(
                abs(xlim[0] - ox),
                abs(xlim[1] - ox),
                abs(ylim[0] - oy),
                abs(ylim[1] - oy),
            )

        for d in self.dirs:
            x = [ox - length * d[0], ox + length * d[0]]
            y = [oy - length * d[1], oy + length * d[1]]
            ax.plot(x, y, **kwargs)
        ax.axis("equal")
        # plt.show()
