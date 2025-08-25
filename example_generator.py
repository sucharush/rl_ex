import numpy as np
from rectangle import Rectangle
from environment import RectangleEnv


def make_cases(base_rect_params, optimizer_params, num_points=10, jitter=0.02):
    """
    Returns a dict[name] -> {
        'rect_params': dict,
        'optimizer_params': dict,
        'points': np.ndarray [num_points, 2],
        'offset': (dx, dy)
    }
    Points are sampled from a rectangle that is the base rectangle shifted by `offset`.
    The env you create later will try to realign these points to the *base* rectangle.
    """
    cases = {}

    def add(
        name, offset=(0.0, 0.0), theta=None, rot=0.0, width=None, height=None, rng=None
    ):
        rect_params = dict(base_rect_params)
        if theta is not None:
            rect_params["theta"] = theta
        if width is not None:
            rect_params["width"] = width
        if height is not None:
            rect_params["height"] = height

        # build a temporary rectangle to generate the points
        R = Rectangle(**rect_params)
        R.move(dx=offset[0], dy=offset[1], theta=rot)
        pts = R.sample_points(num_points=num_points, jitter=jitter, rng=rng)

        cases[name] = {
            "rect_params": rect_params,
            "optimizer_params": dict(optimizer_params),
            "points": pts,
            "offset": tuple(offset),
        }

    rng_ls = [np.random.default_rng(i) for i in range(10)]

    # ---- fixed scenarios ----
    add("aligned", offset=(0.0, 0.0), rot=np.pi / 2, rng=rng_ls[0])
    add("right_1p5", offset=(1.5, 0.0), rot=np.pi / 6, rng=rng_ls[1])
    # add("up_1p0", offset=(0.0, 1.0), rng=rng_ls[2])
    # add("diag_2p0_1p5", offset=(2.0, 1.5), rng=rng_ls[3])
    add("left_down_-1p2_-0p8", offset=(-1.2, -0.8), rot=np.pi / 4, rng=rng_ls[4])
    add(
        "wide_offset_1p0_0p5",
        offset=(1.0, 0.5),
        rot=-np.pi / 4,
        width=3.0,
        rng=rng_ls[5],
    )
    add(
        "tall_offset_-0p5_1p2",
        offset=(-0.5, 1.2),
        rot=-np.pi / 3,
        height=2.0,
        rng=rng_ls[6],
    )
    add(
        "rot_-60_offset_8_-6",
        offset=(8, -6),
        theta=-np.pi / 3,
        rot=-np.pi / 4,
        rng=rng_ls[7],
    )
    add(
        "rot_30_offset_-4_3",
        offset=(-4, 3),
        theta=np.pi / 6,
        rot=np.pi / 2,
        rng=rng_ls[8],
    )
    return cases


def build_envs_from_cases(cases):
    """Convenience: returns dict[name] -> RectangleEnv initialized with that case."""
    envs = {}
    for name, c in cases.items():
        envs[name] = RectangleEnv(
            rect_params=c["rect_params"],
            optimizer_params=c["optimizer_params"],
            points=c["points"],
        )
    return envs
