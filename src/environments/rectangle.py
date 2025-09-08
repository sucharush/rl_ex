import numpy as np
import matplotlib.pyplot as plt


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

    def get_corners(self):
        hw, hh = self.hw, self.hh
        corners = np.array([
            [-hw, -hh],  # bottom-left
            [-hw,  hh],  # top-left
            [ hw,  hh],  # top-right
            [ hw, -hh],  # bottom-right
        ])

        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])

        return corners @ R.T + np.array([self.cx, self.cy])

    def get_sides(self):
        c = self.get_corners()
        return {
            "left": (c[0], c[1]),
            "top": (c[1], c[2]),
            "right": (c[2], c[3]),
            "bottom": (c[3], c[0]),
        }

    def get_local_directions(self):
        return self.horiz, self.vert

    def get_info(self):
        return np.array([self.cx, self.cy, self.theta])

    def move(self, dx=0.0, dy=0.0, theta=0.0):
        self._latest_info = self.get_info()
        self.cx += dx
        self.cy += dy
        self.theta += theta
        self.update_directions()

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        corners = self.get_corners()
        corners = np.vstack([corners, corners[0]])  # close the loop
        ax.plot(corners[:, 0], corners[:, 1], **kwargs)
        ax.set_aspect("equal")


    def points_distance(self, points, distance_only=True):
        """
        Vectorized distance computation: compute distance from many points
        to the rectangle sides without Python loops.
        """
        self.latest_points = points
        points = np.atleast_2d(points)   # shape (N,2)

        # get rectangle corners and 4 segments
        corners = self.get_corners()
        seg_a = np.array([corners[0], corners[1], corners[2], corners[3]])  # (4,2)
        seg_b = np.array([corners[1], corners[2], corners[3], corners[0]])  # (4,2)

        # expand for broadcasting
        P = points[:, None, :]   # shape (N,1,2)
        A = seg_a[None, :, :]    # shape (1,4,2)
        B = seg_b[None, :, :]    # shape (1,4,2)

        AP = P - A               # (N,4,2)
        AB = B - A               # (1,4,2)

        # projection scalar t
        denom = np.sum(AB*AB, axis=-1)   # (1,4)
        t = np.sum(AP*AB, axis=-1) / denom   # (N,4)
        t = np.clip(t, 0.0, 1.0)

        # closest point on segment for each (point, side)
        closest = A + t[..., None] * AB  # (N,4,2)

        # distances
        diff = P - closest
        dist = np.linalg.norm(diff, axis=-1)  # (N,4)

        # choose side with min distance
        idx_min = np.argmin(dist, axis=1)     # (N,)
        min_dist = dist[np.arange(len(points)), idx_min]

        side_names = np.array(["left", "top", "right", "bottom"])
        min_side = side_names[idx_min]

        self._points_distance_ls = min_dist
        self._points_side_ls = min_side.tolist()

        if distance_only:
            return min_dist
        else:
            # collect closest points for each input point
            min_closest = closest[np.arange(len(points)), idx_min]
            results = [(s, d, cp) for s, d, cp in zip(min_side, min_dist, min_closest)]
            return min_dist, results


    def get_mean_dist(self, points):

        return np.mean(self._points_distance_ls)

    def get_distance_list(self, points):

        return self._points_distance_ls

    def get_side_list(self, points):

        return self._points_side_ls

    def sample_points(
        self, offset=(0, 0), rotation=0.0, num_points=10, jitter=0.0, rng=None
    ):
        R = self.copy()
        R.move(dx=offset[0], dy=offset[1], theta=rotation)
        sides = R.get_sides()
        points = []

        if rng is None:
            rng = np.random.default_rng()  # fallback to global

        for _ in range(num_points):
            side_name = rng.choice(list(sides.keys()))
            p1, p2 = sides[side_name]
            alpha = rng.uniform(0, 1)
            point = (1 - alpha) * p1 + alpha * p2
            if jitter > 0:
                point += rng.normal(scale=jitter, size=2)
            points.append(point)

        return np.array(points)

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
    ):
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
            if x.min() < x_lim[0] or x.max() > x_lim[1] or y.min() < y_lim[0] or y.max() > y_lim[1]:
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
            t = np.linspace(0.0, 1.0, k, endpoint=True) if include_corners else np.arange(1, k + 1) / (k + 1)
            base = p1s[i] + t[:, None] * vecs[i]

            if jitter_t > 0:
                jt = rng.normal(scale=jitter_t, size=k)
                base = base + jt[:, None] * tangents[i]
            if jitter_n > 0:
                jn = rng.normal(scale=jitter_n, size=k)
                base = base + jn[:, None] * normals[i]

            pts.append(base)

        return np.vstack(pts) if pts else np.empty((0, 2))

