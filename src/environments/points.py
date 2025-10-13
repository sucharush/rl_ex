import numpy as np
import matplotlib.pyplot as plt

class PointSet:
    def __init__(self, points):
        self.points = np.atleast_2d(np.array(points, dtype=float))
        if self.points.shape[1] != 2:
            raise ValueError("Points must be (N,2) array.")
        self.segments =  np.array(self.build_segments())

    # @property
    # def center(self):
    #     """Centroid of the points."""
    #     return self.points.mean(axis=0)

    # @property
    # def cov(self):
    #     """2x2 covariance matrix of centered points."""
    #     X = self.points - self.center
    #     return X.T @ X / len(self.points)
    # @property
    # def principal_directions(self):
    #     """Eigenvectors sorted by eigenvalue (largest first)."""
    #     eigvals, eigvecs = np.linalg.eigh(self.cov)
    #     order = np.argsort(eigvals)[::-1]
    #     return eigvecs[:, order], eigvals[order]
    # @property
    # def main_direction(self):
    #     """Unit vector of the first principal component."""
    #     dirs, _ = self.principal_directions
    #     return dirs[:, 0]
    # @property
    # def theta(self):
    #     """Angle (radians) of the main direction."""
    #     v = self.main_direction
    #     return np.arctan2(v[1], v[0])

    def plot_points(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.points[:, 0], self.points[:, 1], **kwargs)
        ax.set_aspect("equal")

    def build_segments(self, close_loop: bool = True):
        """Return list of segments as (p1, p2) tuples."""
        segs = [(self.points[i], self.points[i+1])
                for i in range(len(self.points) - 1)]
        if close_loop and len(self.points) > 1:
            segs.append((self.points[-1], self.points[0]))

        # print(len(segs))
        return segs

    def plot(self, close_loop: bool = True, show_points: bool = True, **kwargs):
        """Plot points and (optionally closed) segments."""
        segs = self.build_segments(close_loop=close_loop)

        # plot segments
        for p1, p2 in segs:
             x1, y1 = p1
             x2, y2 = p2
             dx, dy = x2 - x1, y2 - y1
             plt.arrow(x1, y1, dx, dy,
                    head_width=0.08, head_length=0.15,  # adjust to your scale
                    length_includes_head=True,
                    fc="k", ec="k")

        # plot points
        if show_points:
            plt.scatter(self.points[:,0], self.points[:,1], c="red", zorder=5)

        plt.axis("equal")
        # plt.show()

    def resample(self, num_per_segment: int = 10, close_loop: bool = True) -> np.ndarray:
        """
        Resample dense points along each segment.
        Returns (M,2) array with M = num_per_segment * (#segments).
        """
        # =================
        # TODO: length
        # =================
        segs = self.segments
        new_pts = []

        for p1, p2 in segs:
            p1, p2 = np.array(p1), np.array(p2)
            # linspace from p1 to p2, exclude last point to avoid duplicates
            t = np.linspace(0, 1, num_per_segment, endpoint=False)
            interp = (1 - t)[:,None] * p1 + t[:,None] * p2
            new_pts.append(interp)

        return np.vstack(new_pts)
