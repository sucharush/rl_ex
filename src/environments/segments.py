import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Segmentation:
    def __init__(self, nodes: np.ndarray, close_loop: bool = True, num_samples = 50):
        self.nodes = np.atleast_2d(np.array(nodes, dtype=float))
        if self.nodes.shape[1] != 2:
            raise ValueError("Nodes must be (N,2) array.")
        self.close_loop = close_loop
        self._segments = self._build_segments()
        self.points = self.sample(total_points=num_samples)
        self._hull_nodes, self._hull_segments = self._convex_hull()

    @classmethod
    def from_nodes(cls, nodes, close_loop: bool = True, num_samples = 50):
        return cls(nodes, close_loop=close_loop, num_samples = num_samples)

    def _build_segments(self):
        segs = [(self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1)]
        if self.close_loop and len(self.nodes) > 1:
            segs.append((self.nodes[-1], self.nodes[0]))
        return np.array(segs, dtype=float)

    def _convex_hull(self):
        """Return a new Segmentation representing the convex hull of current nodes."""
        hull = ConvexHull(self.nodes)
        hull_nodes = self.nodes[hull.vertices]
        hull_segments = [
            (hull_nodes[i], hull_nodes[(i + 1) % len(hull_nodes)])
            for i in range(len(hull_nodes))
        ]
        return np.array(hull_nodes), np.array(hull_segments)

    @property
    def segments(self):
        return self._segments

    @property
    def hull_nodes(self):
        return self._hull_nodes

    @property
    def hull_segments(self):
        return self._hull_segments

    def sample(self, total_points: int = 100, include_last: bool = False):
        """
        Sample ~total_points distributed proportionally to edge length.

        Parameters
        ----------
        total_points : int
            Desired total number of points (approx; last segment may adjust).
        include_last : bool
            Whether to include the final closing point.
        """
        # lengths of all segments
        seg_lengths = np.linalg.norm(self.segments[:,1] - self.segments[:,0], axis=1)
        total_length = seg_lengths.sum()

        # proportional allocation
        points_per_seg = np.maximum(1, np.floor(total_points * seg_lengths / total_length).astype(int))

        new_pts = []
        for (p1, p2), n in zip(self.segments, points_per_seg):
            # +1 because linspace includes endpoints, but we drop the last
            t = np.linspace(0, 1, n + 1, endpoint=True)
            interp = (1 - t)[:, None] * p1 + t[:, None] * p2
            interp = interp[:-1]  # drop last to avoid duplicates at shared nodes
            new_pts.append(interp)

        pts = np.vstack(new_pts)

        if include_last:
            pts = np.vstack([pts, pts[0]])  # close loop

        self.points = pts
        return self.points

    def plot(
        self,
        ax = None,
        show_nodes: bool = True,
        show_hull: bool = True,
        show_points: bool = False,
        **kwargs
    ):
        """Plot nodes, segments, and optionally sampled points."""
        if ax is None:
            ax = plt.gca()
        # for p1, p2 in self.segments:
        #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="black", **kwargs)
        nodes = np.vstack([self.nodes, self.nodes[0]])  # close loop
        ax.plot(nodes[:, 0], nodes[:, 1], c="black", label="original", **kwargs)

        if show_nodes:
            ax.scatter(
                self.nodes[:, 0],
                self.nodes[:, 1],
                c="red",
                zorder=5,
            )
        if show_points and self.points is not None:
            ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                c="blue",
                s=10,
                zorder=4,
                label="sampled",
            )
        # convex hull overlay
        if show_hull and hasattr(self, "_hull_nodes") and self._hull_nodes is not None:
            hull_pts = self._hull_nodes
            # close the loop
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(
                hull_pts[:, 0],
                hull_pts[:, 1],
                c="gray",
                lw=2,
                ls="--",
                label="convex hull",
            )

        # ax.axis("equal")
        ax.legend()
