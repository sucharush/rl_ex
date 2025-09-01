import numpy as np
from src.environments.rectangle import Rectangle
import matplotlib.pyplot as plt
from src.environments.environment import RectangleEnv
from src.environments.optimizer import StepOptimizer

def test_rectangle_distance():
    """
    Test the distance computation for the Rectangle class.
    """
    rect = Rectangle(center=(0, 0), width=4.0, height=2.0, theta=0.0)

    # Test points exactly on the sides of the rectangle
    points_on_sides = np.array([
        [0, 1],   # Top side
        [0, -1],  # Bottom side
        [2, 0],   # Right side
        [-2, 0],  # Left side
    ])
    # Test points inside the rectangle but not on the sides
    points_inside = np.array([
        [0, 0],    # Center of the rectangle
        [1, 0.5],  # Inside, closer to the top-right
        [-1, -0.5] # Inside, closer to the bottom-left
    ])
    # Test points outside the rectangle
    points_outside = np.array([
        [0, 2],    # Above the top side
        [0, -2],   # Below the bottom side
        [3, 0],    # To the right of the right side
        [-3, 0],   # To the left of the left side
    ])
    # Test points far outside the rectangle
    points_far_outside = np.array([
        [0, 5],    # Far above the top side
        [0, -5],   # Far below the bottom side
        [6, 0],    # Far to the right of the right side
        [-6, 0],   # Far to the left of the left side
    ])
    # Test points diagonally outside the rectangle
    points_diagonal = np.array([
        [3, 2],    # Top-right diagonal
        [-3, 2],   # Top-left diagonal
        [3, -2],   # Bottom-right diagonal
        [-3, -2],  # Bottom-left diagonal
    ])
    points_off_diagonal = np.array([
        [-0.5, 0],    # Diagonal to the right of the rectangle
        [0.5, 0],   # Diagonal to the left of the rectangle

    ])
    fig, ax = plt.subplots()
    rect.plot(ax, color='gray', linestyle='--')
    ax.scatter(points_on_sides[:, 0], points_on_sides[:, 1], color='green', label='On Sides')
    ax.scatter(points_inside[:, 0], points_inside[:, 1], color='blue', label='Inside')
    ax.scatter(points_outside[:, 0], points_outside[:, 1], color='orange', label='Outside')
    ax.scatter(points_far_outside[:, 0], points_far_outside[:, 1], color='purple', label='Far Outside')
    ax.scatter(points_diagonal[:, 0], points_diagonal[:, 1], color='cyan', label='Diagonal Outside')
    ax.scatter(points_off_diagonal[:, 0], points_off_diagonal[:, 1], color='red', label='Off Diagonal')
    ax.legend()
    ax.set_title("Rotated Rectangle with Sampled Contour Points")
    plt.show()

    distances = rect.points_distance(points_on_sides)
    expected_distances = [0.0, 0.0, 0.0, 0.0]
    assert np.allclose(distances, expected_distances), (
        f"Distances for points on sides are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    # Test points exactly on the corners of the rectangle
    corners = rect.get_corners()
    distances = rect.points_distance(corners)
    expected_distances = [0.0, 0.0, 0.0, 0.0]
    assert np.allclose(distances, expected_distances), (
        f"Distances for points on corners are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )
    # Test points inside the rectangle
    distances = rect.points_distance(points_inside)
    expected_distances = [1.0, 0.5, 0.5]
    assert np.allclose(distances, expected_distances), (
        f"Distances for points inside the rectangle are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    
    distances = rect.points_distance(points_outside)
    expected_distances = [1.0, 1.0, 1.0, 1.0]  # All points are 1 unit away from the closest side
    assert np.allclose(distances, expected_distances), (
        f"Distances for points outside the rectangle are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    
    distances = rect.points_distance(points_far_outside)
    expected_distances = [4.0, 4.0, 4.0, 4.0]  # All points are 4 units away from the closest side
    assert np.allclose(distances, expected_distances), (
        f"Distances for far points outside the rectangle are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    
    distances = rect.points_distance(points_diagonal)
    expected_distances = [np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]  # Diagonal distances
    assert np.allclose(distances, expected_distances), (
        f"Distances for diagonally outside points are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    distances = rect.points_distance(points_off_diagonal)
    expected_distances = [1, 1]  
    assert np.allclose(distances, expected_distances), (
        f"Distances for off diagonal points are incorrect. "
        f"Computed: {distances}, Expected: {expected_distances}"
    )

    print("All tests passed!")

# Run the test
# test_rectangle_distance()

def sanity_check_step_optimizer():
    """
    Sanity check for StepOptimizer and Rectangle classes.
    Verifies that points are iteratively adjusted to minimize their distance to the rectangle.
    """
    rect = Rectangle(center=(0, 0), width=2.0, height=1.0, theta=0.0)
    _, ax = plt.subplots()
    rect.plot(ax=ax, color='black', linestyle='-')
    
    # Initial points
    points = np.array([[3.0, 1.0], [3.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    ax.scatter(points[:, 0], points[:, 1], color="red", label="Initial Points")
    
    # Optimizer setup
    optimizer = StepOptimizer(loss="soft_l1", bounds=(-3, 3))
    
    max_iter = 0
    while rect.get_mean_dist(points) >= 0.01 and max_iter <= 20:
        used1, delta1 = optimizer.run(
            rect=rect, points=points, direction="horizontal", max_nfev=100
        )
        # points[:, 0] += delta1
        rect.move(dx=delta1, dy=0.0, theta=0.0)
        max_iter += 1
        print(f"Step {max_iter}: horizontal, shift={delta1}, used_iters={used1}")

        used2, delta2 = optimizer.run(
            rect=rect, points=points, direction="vertical", max_nfev=100
        )
        max_iter += 1
        rect.move(dx=0.0, dy=delta2, theta=0.0)
        print(f"Step {max_iter}: vertical, shift={delta2}, used_iters={used2}")
        # points[:, 1] += delta2

    # Final points
    ax.scatter(points[:, 0], points[:, 1], color="red", marker="o", label="Points")
    rect.plot(ax=ax, color='gray', linestyle = '--', label='Final Rectangle')
    ax.legend()
    plt.show()
# sanity_check_step_optimizer()

# ----------------------- Helpers -----------------------

def make_points_sample():
    """A tiny cloud mostly to the right/top of the rectangle."""
    return np.array([[3.0, 1.0],
                     [3.0, 0.0],
                     [2.0, 0.0],
                     [1.0, 1.0]], dtype=float)

def build_env():
    rect_params = dict(center=(0, 0), width=2.0, height=1.0, theta=0.0)
    optimizer_params = dict(loss="soft_l1", bounds=(-5, 5))
    pts = make_points_sample()
    env = RectangleEnv(rect_params=rect_params,
                       optimizer_params=optimizer_params,
                       points=pts)
    
    return env

def print_state_tuple(s):
    (num_top, num_bottom, num_right, num_left,
     num_touching, num_close, num_far) = s
    print(f" sides    (T,B,R,L): {num_top:2d} {num_bottom:2d} {num_right:2d} {num_left:2d}")
    print(f" distances (touch,close,far): {num_touching:2d} {num_close:2d} {num_far:2d}")

# ----------------------- Checks -----------------------

def check_encode_state_invariants(env):
    print("\n[check] encode_state invariants")
    s = env.encode_state()
    print(f"Encoded state: {s}")
    N = len(env.points)
    print_state_tuple(s)
    a, b, c, d, e, f, g = s
    ok1 = (a + b + c + d) == N
    ok2 = (e + f + g) == N
    print(f"  sum sides == N: {ok1}   sum bins == N: {ok2}")

def check_axis_isolation(env):
    print("\n[check] axis isolation via apply_action()")
    env.reset()
    x0 = env.points[:, 0].copy()
    y0 = env.points[:, 1].copy()
    env.apply_action(0)  # ("vertical", 2)
    print(f"  vertical: x unchanged? {np.allclose(env.points[:,0], x0)} | y changed? {not np.allclose(env.points[:,1], y0)}")

    env.reset()
    x0 = env.points[:, 0].copy()
    y0 = env.points[:, 1].copy()
    env.apply_action(3)  # ("horizontal", 2)
    print(f"  horizontal: y unchanged? {np.allclose(env.points[:,1], y0)} | x changed? {not np.allclose(env.points[:,0], x0)}")

def rollout_and_print(env, steps=20):
    print("\n[run] short rollout using env.step()")
    env.reset()
    # Compute initial mean distance
    dist_prev = env.rectangle.get_mean_dist(points=env.points)
    print(f"{'t':>2}  {'action':>12}  {'used':>4}  {'mean_before':>12}  {'mean_after':>11}  {'imprv':>7}  {'reward':>9}")
    print("-"*72)

    # Cycle through all 18 actions to exercise budgets/directions
    action_order = [3, 0, 4, 1, 5, 2, 6, 7, 8, 9, 10, 13, 17, 12, 14, 15, 2, 5, 16]  # H2, V2, H4, V4, H6, V6
    for t in range(steps):
        a = action_order[t % len(action_order)]
        direction, budget = env.actions[a]

        # Step once
        state, reward, done = env.step(a)
        # After step(), env._old_dist should reflect the new mean; recompute to be robust
        dist_after = env.rectangle.get_mean_dist(points=env.points)
        used = env.latest_used_iters
        improvement = dist_prev - dist_after

        print(f"{t:2d}  {str((direction,budget)):>12}  {used:4d}  {dist_prev:12.6f}  {dist_after:11.6f}  {improvement:7.4f}  {reward:9.4f}")

        # Optional soft check (printed, not asserted): expected reward if alpha=1, beta=0.1
        expected = improvement - 0.1 * used
        print(f"      expected: {expected:.4f}  (set compute_reward(alpha=1,beta=0.1) to match)")

        # Invariants each step
        a1, a2, a3, a4, b1, b2, b3 = state
        N = len(env.points)
        inv1 = (a1 + a2 + a3 + a4) == N
        inv2 = (b1 + b2 + b3) == N
        print(f"      invariants: sides_sum==N? {inv1} | bins_sum==N? {inv2}")

        dist_prev = dist_after
        if done:
            print("  -> done (terminal reached by touching-all or max_steps)")
            break

def check_terminal_conditions(env):
    print("\n[check] terminal conditions")
    # Case: all points exactly on the right edge
    env.reset()
    N = len(env.points)
    env.points[:] = np.array([env.rectangle.x_right if hasattr(env.rectangle, 'x_right') else 1.0,
                              0.0])[None, :].repeat(N, axis=0)
    print(f"  all-touching? {env.is_terminal()}  (expect True if your tau <= tiny epsilon)")

    # Case: max steps
    env.reset()
    env.steps = env.max_steps
    print(f"  max-steps terminal? {env.is_terminal()}  (expect True)")

# # ----------------------- Main -----------------------

# if __name__ == "__main__":
#     # Run the sanity checks for the Rectangle class
#     test_rectangle_distance()
#     # Run the sanity checks for the RectangleEnv
#     sanity_check_step_optimizer()

#     env = build_env()
#     env.render()

#     # 1) state invariants on reset
#     env.reset()
#     check_encode_state_invariants(env)

#     # 2) axis isolation (uses apply_action only)
#     check_axis_isolation(env)

#     # 3) rollout that prints distances, rewards, and invariants
#     rollout_and_print(env, steps=18)

#     # 4) terminal checks
#     check_terminal_conditions(env)

#     print("\n[done] If something looks off (e.g., reward != expected), tweak compute_reward or the alpha/beta used.")