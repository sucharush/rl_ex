import numpy as np
from environments.rectangle import Rectangle
from environments.environment import RectangleEnv

def make_cases(base_rect_params, optimizer_params,
               num_points=10, jitter_t = 0.05, jitter_n=0.01,
               offsets=None, rotations=None, rng=None):
    """
    Returns dict[name] -> case with:
      - fixed base rectangle shape
      - points generated from that shape displaced/rotated
    """

    if rng is None:
        rng = np.random.default_rng()

    cases = {}
    rect_params = dict(base_rect_params)

    # default sets if none provided
    if offsets is None:
        offsets = [(0,0), (1.5,0), (-1.2,-0.8), (2,1), (-3,2)]
    if rotations is None:
        rotations = [0.0, np.pi/6, np.pi/4, -np.pi/4, np.pi/2]

    for i,(dx,dy) in enumerate(offsets):
        for j,rot in enumerate(rotations):
            R = Rectangle(**rect_params)
            R.move(dx=dx, dy=dy, theta=rot)
            pts = R.sample(num_points=num_points, jitter_t=jitter_t, jitter_n=jitter_n, rng=rng)

            name = f"offset_{dx:.1f}_{dy:.1f}_rot_{np.degrees(rot):.0f}"
            cases[name] = {
                "rect_params": dict(rect_params),          # same base rect
                "optimizer_params": dict(optimizer_params),
                "points": pts,
                "offset": (dx,dy),
                "rot": rot,
            }

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