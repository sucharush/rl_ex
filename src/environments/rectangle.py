import numpy as np
import matplotlib.pyplot as plt
# from src.environments.points import PointSet
# from src.environments.rays import RayBatch
from src.environments.segments import Segmentation
from src.utils.geometry import (
    rays_segments_intersection,
    rays_segments_span,
    prependicular,
    point_inside_segments,
    normalize_angle,
)
from typing import Optional


class Rectangle:
    def __init__(self, center=(0.0, 0.0), width=2.0, height=1.0, theta=0.0):
        self.cx, self.cy = center
        self.hw = width / 2
        self.hh = height / 2
        self.theta = theta  # in radians, counter-clockwise from the positive x-axis
        self.latest_points = None
        self._points_distance_ls = None
        self._points_side_ls = None
        self._latest_info = np.array([self.cx, self.cy, self.theta])
        self.update_directions()

    def update_directions(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.horiz = np.array([c, s])  # local x in world
        self.vert = np.array([-s, c])  # local y in world

    def copy(self):
        return Rectangle(
            center=(self.cx, self.cy),
            width=self.hw * 2,
            height=self.hh * 2,
            theta=self.theta,
        )

    def get_direction(self):
        return np.array([self.horiz, self.vert])

    def get_corners(self):
        # ccw!!!
        hw, hh = self.hw, self.hh

        corners = np.array([
            [-hw, -hh],  # bottom-left
            [ hw, -hh],  # bottom-right
            [ hw,  hh],  # top-right
            [-hw,  hh],  # top-left
        ])

        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        return corners @ R.T + np.array([self.cx, self.cy])

    def get_sides(self):
        # ccw!!!
        c = self.get_corners()
        return {
            "left": (c[3], c[0]),
            "bottom": (c[0], c[1]),
            "right": (c[1], c[2]),
            "top": (c[2], c[3]),
        }

    def get_contour(self, ccw=True):
        c = self.get_corners()
        c_next = np.roll(c, 1, axis=0)
        c = c[:, None]
        c_next = c_next[:, None]
        if ccw:
            # l, t, r, b
            contour = np.hstack((c_next, c))
        else:
            contour = np.hstack((c, c_next))
        return contour

    def get_center(self):
        center = np.array([self.cx, self.cy])
        return center

    def get_local_directions(self):
        return self.horiz, self.vert

    def get_info(self):
        return np.array([self.cx, self.cy, self.theta])

    # def get_mean_dist(self, points):

    #     return np.mean(self._points_distance_ls)

    # def get_distance_list(self, points):

    #     return self._points_distance_ls

    # def get_side_list(self, points):

    #     return self._points_side_ls

    def move(self, dx=0.0, dy=0.0, theta=0.0):
        self._latest_info = self.get_info()
        self.cx += dx
        self.cy += dy
        self.theta += theta
        self.theta = normalize_angle(self.theta)
        self.update_directions()

    def move_to(self, cx, cy, theta):
        self._latest_info = self.get_info()
        self.cx, self.cy, self.theta = cx, cy, theta
        self.update_directions()

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        corners = self.get_corners()
        corners = np.vstack([corners, corners[0]])  # close the loop
        ax.plot(corners[:, 0], corners[:, 1], **kwargs)
        ax.set_aspect("equal")

    def points_signed_segment_distance(self, points):
        """
        Signed distance from points to each rectangle edge segment.
        Positive = outside (in outward normal direction),
        Negative = inside.
        """
        points = np.atleast_2d(points)  # (N,2)

        # rectangle corners (already rotated by theta)
        corners = self.get_corners()
        seg_a = np.array([corners[0], corners[1], corners[2], corners[3]])  # (4,2)
        seg_b = np.array([corners[1], corners[2], corners[3], corners[0]])  # (4,2)
        AB = seg_b - seg_a  # (4,2)

        # # outward normals for CCW: (dy, -dx)
        # normals = np.stack([AB[:, 1], -AB[:, 0]], axis=1)
        # normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals = prependicular(AB)

        # broadcasting
        P = points[:, None, :]  # (N,1,2)
        A = seg_a[None, :, :]  # (1,4,2)
        # B = seg_b[None, :, :]    # (1,4,2)
        AB = AB[None, :, :]  # (1,4,2)

        # projection scalar along edge, clamped to [0,1]
        t_raw = np.sum((P - A) * AB, axis=-1) / np.sum(AB * AB, axis=-1)  # (N,4)
        t = np.clip(t_raw, 0.0, 1.0)

        # closest point on the segment
        closest = A + t[..., None] * AB  # (N,4,2)

        # Euclidean distance to the segment
        diff = P - closest
        dist = np.linalg.norm(diff, axis=-1)  # (N,4)

        # signed by supporting line (same normals as before)
        signs = np.sign(np.sum((P - A) * normals[None, :, :], axis=-1))  # (N,4)
        signs[signs == 0] = 1 # override with 1 when the points are on the extension of a side
        signed_dist = signs * dist

        side_names = np.array([ "bottom", "right", "top", "left"])
        return signed_dist, side_names

    def points_signed_distance(self, points: np.ndarray) -> np.ndarray:
        distances, _ = self.points_signed_segment_distance(points=points)
        idx = np.argmin(abs(distances), axis=1)
        dist_list = distances[np.arange(distances.shape[0]), idx]
        # print(dist_list)
        return dist_list

    def points_distance(self, points: np.ndarray) -> np.ndarray:
        distances, _ = self.points_signed_segment_distance(points=points)
        idx = np.argmin(abs(distances), axis=1)
        dist_list = np.abs(distances[np.arange(distances.shape[0]), idx])
        # print(dist_list)
        return dist_list

    def points_square_root_dist(self, points: np.ndarray) -> np.ndarray:
        distances, _ = self.points_signed_segment_distance(points=points)
        idx = np.argmin(abs(distances), axis=1)
        dist_list = np.mean(distances[np.arange(distances.shape[0]), idx]**2)
        return dist_list

    def sample(
        self,
        num_points=10,
        offset=(0, 0),
        rotation=0.0,
        jitter_t=0.0,
        jitter_n=0.0,
        rng=None,
        include_corners=False,
        x_lim=None,
        y_lim=None,
        validate=False,
    ) -> np.ndarray:
        """
        Deterministic, near-uniform sampling along the rectangle perimeter.
        - counts per side according to side length
        - evenly spaced along each side
        - optional jitter: along-edge (jitter_t) and normal (jitter_n)
        - if validate=True, skip sampling if rectangle corners exceed x_lim / y_lim
        """
        R = self.copy()
        R.move(dx=offset[0], dy=offset[1], theta=rotation)
        if rng is None:
            rng = np.random.default_rng()

        # --- optional bounds check ---
        if validate and (x_lim is not None and y_lim is not None):
            x, y = R.get_corners().T
            if (
                x.min() < x_lim[0]
                or x.max() > x_lim[1]
                or y.min() < y_lim[0]
                or y.max() > y_lim[1]
            ):
                print("Warning >>> Skipping invalid rectangle...")
                return None

        # Build edges in world coords (p1->p2)
        sides = list(R.get_sides().values())
        p1s = np.stack([p1 for (p1, _) in sides], axis=0)
        p2s = np.stack([p2 for (_, p2) in sides], axis=0)

        vecs = p2s - p1s
        lengths = np.linalg.norm(vecs, axis=1).astype(float)
        total = lengths.sum()

        if total == 0:
            return np.empty((0, 2))

        # Allocate counts ∝ length
        raw = num_points * lengths / total
        counts = np.floor(raw).astype(int)
        remainder = num_points - counts.sum()
        if remainder > 0:
            order = np.argsort(raw - counts)[::-1]
            counts[order[:remainder]] += 1

        # Unit tangents and normals
        tangents = np.divide(vecs, lengths[:, None], where=lengths[:, None] > 0)
        normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)

        pts = []
        for i, k in enumerate(counts):
            if k <= 0:
                continue
            t = (
                np.linspace(0.0, 1.0, k, endpoint=True)
                if include_corners
                else np.arange(1, k + 1) / (k + 1)
            )
            # also in clockwise!!!
            base = p1s[i] + t[:, None] * vecs[i]

            if jitter_t > 0:
                jt = rng.normal(scale=jitter_t, size=k)
                base = base + jt[:, None] * tangents[i]
            if jitter_n > 0:
                jn = rng.normal(scale=jitter_n, size=k)
                base = base + jn[:, None] * normals[i]

            pts.append(base)
        return np.vstack(pts) if pts else np.empty((0, 2))

    def angle_span_from_points(self, segs: Segmentation):
        center = self.get_center()
        theta_base = self.theta
        inside = point_inside_segments(o=center, segments=segs.hull_segments)
        if inside:
            # print("is inside")
            theta_min, theta_max = self.get_default_range()
            return theta_min, theta_max, inside

        R = self.get_direction()
        rel_local = (segs.hull_nodes - center) @ R.T
        thetas = np.arctan2(rel_local[:, 1], rel_local[:, 0])  # [-pi, pi]

        thetas = np.mod(thetas, 2 * np.pi)
        thetas_sorted = np.sort(thetas)

        # this only works when the polygon is not too concave

        diffs = np.diff(np.r_[thetas_sorted, thetas_sorted[0] + 2 * np.pi])
        cut_idx = np.argmax(diffs)
        max_gap = diffs[cut_idx]

        theta_min = thetas_sorted[(cut_idx + 1) % len(thetas_sorted)]
        theta_max = thetas_sorted[cut_idx]

        # Normalize relative to rectangle orientation
        theta_min = np.mod(theta_min + theta_base, 2 * np.pi)
        theta_max = np.mod(theta_max + theta_base, 2 * np.pi)
        return theta_min, theta_max, inside

    def get_default_range(self):
        theta_min = normalize_angle(self.theta)
        theta_max = normalize_angle(self.theta + np.pi)
        return theta_min, theta_max

    def internal_dist(self, dirs):
        o = self.get_center()
        contour = self.get_contour()
        p1s, p2s = contour[:, 0], contour[:, 1]
        distances = rays_segments_span(o=o, dirs=dirs, p1s=p1s, p2s=p2s)
        _, t_max = distances
        return t_max
