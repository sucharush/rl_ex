import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from environments.environment import RectangleEnv


# ----- policy from Q -----
def epsilon_greedy(Q, s, eps, nA):
    if np.random.random() < eps:
        return np.random.randint(nA)
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(np.random.choice(best))


def greedy_action(Q, s):
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(np.random.choice(best))

# def greedy_action(Q, s, N=None, unvisited_penalty=1e-2):
#     q = Q[s].copy()
#     if N is not None:
#         # mark unvisited with slight penalty
#         unvisited = (N[s] == 0)
#         q[unvisited] -= unvisited_penalty
#     best = np.flatnonzero(q == q.max())
#     return int(np.random.choice(best))

# ----- render control ----- 
@contextmanager
def suppress_render(env):
    orig = env.render
    env.render = lambda *a, **k: None
    try:
        yield
    finally:
        env.render = orig


# ----- SARSA training -----
def train_sarsa(
    env=RectangleEnv,
    episodes=300,
    alpha=0.1,
    gamma=1,
    eps_start=0.2,
    eps_end=0.01,
    eps_decay=0.995,
    seed=0,
):
    """
    Returns:
      Q: defaultdict[state_tuple] -> np.array(nA)
      logs: list of dicts with per-episode metrics:
            {'return', 'steps', 'iters', 'solved', 'final_mean_dist', 'eps'}
    """
    np.random.seed(seed)
    nA = len(env.actions)
    Q = defaultdict(lambda: np.zeros(nA, dtype=float))
    N = defaultdict(lambda: np.zeros(nA, dtype=int))
    logs = []

    with suppress_render(env):  # no pop-ups while learning
        for ep in range(episodes):
            s = env.reset()
            eps = max(eps_end, eps_start * (eps_decay**ep))
            a = epsilon_greedy(Q, s, eps, nA)

            G = 0.0
            total_used_iters = 0
            steps = 0
            done = False

            while not done:
                s2, r, done = env.step(a)
                G += r
                steps += 1
                # accumulate optimizer iterations used this step
                total_used_iters += int(env.latest_used_iters or 0)

                if done:
                    target = r
                else:
                    a2 = epsilon_greedy(Q, s2, eps, nA)
                    if env.is_stalled():
                        a2 = nA - 1
                        # print(f"action: {env.actions[a2]}")
                    target = r + gamma * Q[s2][a2]

                Q[s][a] += alpha * (target - Q[s][a])
                s, a = s2, (a2 if not done else a)

            # episode summary
            final_mean_dist = env.rectangle.get_mean_dist(points=env.points)
            solved = bool(
                np.all(
                    np.array(env.rectangle.points_distance(points=env.points)) < env.tau
                )
            )
            logs.append(
                {
                    "return": G,
                    "steps": steps,
                    "iters": total_used_iters,
                    "solved": solved,
                    "final_mean_dist": float(final_mean_dist),
                    "eps": eps,
                }
            )

    return Q, logs, N


# ----- Q-learning training -----
def train_q_learning(
    env=RectangleEnv,
    episodes=300,
    alpha=0.1,
    gamma=1,
    eps_start=0.2,
    eps_end=0.01,
    eps_decay=0.995,
    seed=0,
):
    """
    Returns:
      Q: defaultdict[state_tuple] -> np.array(nA)
      logs: list of dicts with per-episode metrics:
            {'return', 'steps', 'iters', 'solved', 'final_mean_dist', 'eps'}
    """
    np.random.seed(seed)
    nA = len(env.actions)
    Q = defaultdict(lambda: np.zeros(nA, dtype=float))
    # Q = defaultdict(lambda: np.full(nA, 0.1, dtype=float))
    N = defaultdict(lambda: np.zeros(nA, dtype=int))
    logs = []

    with suppress_render(env):  # no pop-ups while learning
        for ep in range(episodes):
            s = env.reset()
            eps = max(eps_end, eps_start * (eps_decay**ep))
            a = epsilon_greedy(Q, s, eps, nA)

            G = 0.0
            total_used_iters = 0
            steps = 0
            done = False

            while not done:
                # 1) greedly choose action
                a = epsilon_greedy(Q, s, eps, nA)
                # if env.is_stalled():
                #     a = nA - 1
                    # print(f"Stalled! action: {env.actions[a]}")
                    # print(a)
                # a = epsilon_greedy(Q, s, eps, nA)
                # print(f"action: {env.actions[a]}")
                # 2) step in env
                N[s][a] += 1

                s2, r, done = env.step(a)
                G += r
                steps += 1
                total_used_iters += int(env.latest_used_iters or 0)

                # 3) TD update
                if done:
                    target = r
                else:
                    target = r + gamma * np.max(Q[s2])  # max over all actions in s2

                # 4) update Q-value
                Q[s][a] += alpha * (target - Q[s][a])

                s = s2

            # episode summary
            final_mean_dist = env.rectangle.get_mean_dist(points=env.points)
            solved = bool(
                np.all(
                    np.array(env.rectangle.points_distance(points=env.points)) < env.tau
                )
            )
            logs.append(
                {
                    "return": G,
                    "steps": steps,
                    "iters": total_used_iters,
                    "solved": solved,
                    "final_mean_dist": float(final_mean_dist),
                    "eps": eps,
                }
            )

    return Q, logs, N


def run_episode(env=RectangleEnv, Q=defaultdict, N = defaultdict, max_steps=100, render=True):
    """
    Runs one episode with greedy policy (ε=0), including stall fallback.
    Returns dict: {'return','steps','iters','solved','final_mean_dist'}
    """

    # choose whether to suppress rendering
    class _Null:
        def __enter__(self): pass
        def __exit__(self, *args): pass

    ctx = _Null() if render else suppress_render(env)

    with ctx:
        s = env.reset()
        if render:
            env.render()

        total = 0.0
        total_used_iters = 0
        steps = 0
        force_next = False  # NEW

        for _ in range(max_steps):
            if force_next:
                a = len(env.actions) - 1   # force manual rotate
                force_next = False
            else:
                # a = greedy_action(Q, s, N=N)
                a = greedy_action(Q, s)

            s2, r, done = env.step(a)
            total += r
            steps += 1
            total_used_iters += int(env.latest_used_iters or 0)

            # schedule fallback for next action
            if not done and env.is_stalled():
                force_next = True

            s = s2
            if done:
                break

    final_mean_dist = env.rectangle.get_mean_dist(points=env.points)
    solved = bool(
        np.all(np.array(env.rectangle.points_distance(points=env.points)) < env.tau)
    )
    return {
        "return": total,
        "steps": steps,
        "iters": total_used_iters,
        "solved": solved,
        "final_mean_dist": float(final_mean_dist),
    }

