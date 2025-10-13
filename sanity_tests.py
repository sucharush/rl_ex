from src.environments.rectangle import Rectangle  # adjust import
from src.environments.segments import Segmentation
from src.environments.environment import RectangleEnv
import numpy as np


def sanity_test_points_signed_distance():

    rect = Rectangle(
        center=(0, 0), width=4 * np.sqrt(2), height=2 * np.sqrt(2), theta=np.pi / 4
    )

    points = np.array(
        [
            (3, 3),
            (3, 1),
            (0, 0),
            (-3, 3),
            (3, 5),
            (2, 2),
        ]
    )

    signed, _ = rect.points_signed_segment_distance(points)
    # print(signed)

    expected = np.array(
        [
            [-2.0, +np.sqrt(2), -2.0, -5 * np.sqrt(2)],
            [0.0, 0.0, -2 * np.sqrt(2), -4 * np.sqrt(2)],
            [-np.sqrt(2), -2 * np.sqrt(2), -np.sqrt(2), -2 * np.sqrt(2)],
            [-4 * np.sqrt(2), -4.0, 2 * np.sqrt(2), -4.0],
            [-4, 2 * np.sqrt(2), 2 * np.sqrt(2), -6 * np.sqrt(2)],
            [-np.sqrt(2), 0, -np.sqrt(2), -4 * np.sqrt(2)],
        ]
    )
    np.testing.assert_allclose(signed, expected, atol=1e-8)
    print("All samples passed.\n")


def sanity_test_env():
    """
    Run sanity checks on RectangleEnv given a specific center.
    Prints results of the invariants we expect.
    """
    test_cases = [
        ((0, 3), {"center_inside": False, "case": "A"}),  # expect all positives
        ((4, 0), {"center_inside": False, "case": "B"}),  # expect half zeros, half ≥0
        ((1, 0), {"center_inside": True, "case": "C"}),  # expect symmetry
    ]
    n_rays = np.random.choice([2, 3, 5,])
    dist_len = 2*n_rays
    optimizer_params = dict(loss="soft_l1", bounds=(-5, 5))
    rays_params = dict(n_rays=n_rays)
    rect_params = dict(center=(0, 0), width=4.0, height=2.0, theta=0)

    for move_center, expect in test_cases:
        print(f"Case {expect['case']}: moved center={move_center}, center_inside={expect['center_inside']}")
        # reset rect at (0,0), then move
        R = Rectangle(**rect_params)
        R.move(dx=move_center[0], dy=move_center[1])

        # rebuild segmentation on moved rectangle
        segs = Segmentation.from_nodes(R.get_corners())
        env = RectangleEnv(rect_params, optimizer_params, rays_params, segs)

        state = env.state
        ci = env.rays.center_inside

        if expect["case"] == "A":
            # env.render(show=True)
            assert not ci, "Expected center_inside=False"
            assert np.all(state[:dist_len] > 0), f"Expected state[:2n] > 0; got {state[:dist_len]}"
            env.step(action=0)

        elif expect["case"] == "B":
            # env.render(show=True)
            assert not ci, "Expected center_inside=False"
            assert np.allclose(state[:n_rays], 0.0), f"Expected state[:n] = 0; got {state[:n_rays]}"
            assert np.all(state[n_rays:dist_len] >= 0.0), f"Expected state[n:2n] >= 0; got {state[n_rays:dist_len]}"
            env.step(action=1)
            env.step(action=1)

        elif expect["case"] == "C":
            # env.render(show=True)
            assert ci, "Expected center_inside=True"
            # print(state)
            # print(env.rays.thetas)
            # print(env.rays.default_dist)
            assert np.allclose(
                state[:n_rays], state[n_rays:dist_len][::-1], atol=1e-8
            ), f"Expected state[:n] = -state[n:2n][::-1]; got {state[:n_rays]} vs {state[n_rays:dist_len][::-1]}"
            # print(state[:n_rays] - (-state[n_rays:dist_len][::-1]))
            env.step(action=1)
        assert np.allclose(
            env.state[:dist_len], 0.0, atol=0.1
        ), "Distance is not close after applying the optimizer."
        env.reset()
        assert np.allclose(env.state, state), "The state does not reset correctly."
        print(" Passed checks.")

    print("All samples passed.")
