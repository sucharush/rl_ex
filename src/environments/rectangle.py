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
        self._latest_info = (center, theta)
        self.update_directions()

    def get_local_directions(self):
        """
        Return unit vectors (horizontal, vertical) in world coordinates
        according to the current orientation theta.
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        # rotation matrix
        R = np.array([[c, -s], [s, c]])
        horiz = R @ np.array([1.0, 0.0])
        vert = R @ np.array([0.0, 1.0])
        return horiz, vert
    

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

    def move(self, dx=0.0, dy=0.0, theta=0.0):
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

    @staticmethod
    def _cross(v1, v2):
        """
        Cross product helper function
        +: counter-clockwise, -: clockwise, 0: collinear
        """
        return v1[0] * v2[1] - v1[1] * v2[0]

    @staticmethod
    def _point_to_segment_distance_with_closest(p, a, b):
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest

    def _point_distance_to_all_sides(self, point):
        """
        Calculate distance from point to all four sides of rectangle
        Returns the minimum distance and corresponding side
        """
        sides = self.get_sides()
        distances = {}
        closest_points = {}

        # Calculate distance to each side
        for side_name, (a, b) in sides.items():
            dist, closest = self._point_to_segment_distance_with_closest(
                np.array(point), a, b
            )
            distances[side_name] = dist
            closest_points[side_name] = closest

        # Find minimum distance
        min_side = min(distances, key=distances.get)
        min_distance = distances[min_side]
        closest_point = closest_points[min_side]
        return min_side, min_distance, closest_point

    def points_distance(self, points, distance_only=True):
        """
        Improved version that computes distance to nearest side correctly
        """
        self.latest_points = points
        points = np.atleast_2d(points)

        results = []

        for point in points:
            side, dist, closest_point = self._point_distance_to_all_sides(point)
            results.append((side, dist, closest_point))

        distances = np.array([r[1] for r in results])
        sides = [r[0] for r in results]

        self._points_distance_ls = distances
        self._points_side_ls = sides

        if distance_only:
            return distances
        else:
            return distances, results

    def get_mean_dist(self, points):
        # print("Latest points:", self.latest_points, type(self.latest_points))
        # print("Current points:", points, type(points))
        if (
            self.latest_points is None
            or not np.array_equal(np.array(self.latest_points), np.array(points))
            or not self._latest_info == ((self.cx, self.cy), self.theta)
        ):
            # print("Computing distances for new points")
            self.points_distance(points)
        return np.mean(self._points_distance_ls)

    def get_distance_list(self, points):
        if (
            self.latest_points is None
            or not np.array_equal(np.array(self.latest_points), np.array(points))
            or not self._latest_info == ((self.cx, self.cy), self.theta)
        ):
            self.points_distance(points)
        return self._points_distance_ls

    def get_side_list(self, points):

        if (
            self.latest_points is None
            or not np.array_equal(np.array(self.latest_points), np.array(points))
            or not self._latest_info == ((self.cx, self.cy), self.theta)
        ):
            self.points_distance(points)
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

