from src.environments.environment_old import RectangleEnv
import numpy as np
from src.environments.rectangle import Rectangle
from src.environments.segments import Segmentation
# from src.models.agent_ann_base import QNetwork

def compute_safe_limits(rect_params, offset_range):
    """
    Compute world limits (x_lim, y_lim) big enough to contain the rectangle
    under all allowed offsets and rotations.

    rect_params: dict with 'width' and 'height'
    offset_range: ((dx_min, dx_max), (dy_min, dy_max))

    Returns:
        x_lim, y_lim  (tuples)
    """
    hw = rect_params["width"] / 2
    hh = rect_params["height"] / 2

    # worst-case half-extent under arbitrary rotation
    max_extent = np.sqrt(hw**2 + hh**2)

    dx_min, dx_max = offset_range[0]
    dy_min, dy_max = offset_range[1]

    # take absolute largest offsets in each direction
    max_dx = max(abs(dx_min), abs(dx_max))
    max_dy = max(abs(dy_min), abs(dy_max))

    x_lim = (int(np.floor(-(max_extent + max_dx))), int(np.ceil(max_extent + max_dx)))
    y_lim = (int(np.floor(-(max_extent + max_dy))), int(np.ceil(max_extent + max_dy)))

    return x_lim, y_lim


def build_env(rect_params, optimizer_params, rays_params, segs, x_lim, y_lim):
    R = Rectangle(**rect_params)
    # points = R.sample(**rays_params)
    # if points is None or len(points) == 0:
    #     return None
    return RectangleEnv(
        rect_params=rect_params,
        optimizer_params=optimizer_params,
        rays_params = rays_params,
        segs=segs,
        x_lim=x_lim,
        y_lim=y_lim,
    )


def make_env_generator(
    rect_params,
    optimizer_params,
    rays_params,
    offset_range=((-3, 3), (-3, 3)),
    rotation_range=(-np.pi / 2, np.pi / 2),
    rng=None,
    x_lim=None,
    y_lim=None,
):
    """
    Returns a generator function that creates RectangleEnv instances
    with random offset/rotation applied to the base points.
    """

    rng = rng or np.random.default_rng()

    # ensure user-provided limits are large enough
    safe_x, safe_y = compute_safe_limits(rect_params, offset_range)
    if x_lim is None:
        x_lim = safe_x
    else:
        x_lim = (min(x_lim[0], safe_x[0]), max(x_lim[1], safe_x[1]))
    if y_lim is None:
        y_lim = safe_y
    else:
        y_lim = (min(y_lim[0], safe_y[0]), max(y_lim[1], safe_y[1]))
    print(f"Using x_lim={x_lim}, y_lim={y_lim} to accommodate offsets/rotations.")

    def _generator():
        dx = rng.uniform(*offset_range[0])
        dy = rng.uniform(*offset_range[1])
        rot = rng.uniform(*rotation_range)
        R = Rectangle(**rect_params)
        R.move(dx=dx, dy=dy, theta=rot)
        segs = Segmentation.from_nodes(R.get_corners())

        # pp = dict(base_points_params)
        # pp["offset"] = (dx, dy)
        # pp["rotation"] = rot

        return build_env(rect_params, optimizer_params, rays_params, segs, x_lim, y_lim)

    return _generator
