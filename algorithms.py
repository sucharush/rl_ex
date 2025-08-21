import numpy as np
from collections import defaultdict
from contextlib import contextmanager

# ----- policy from Q -----
def epsilon_greedy(Q, s, eps, nA):
    if np.random.random() < eps:
        return np.random.randint(nA)
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(np.random.choice(best))

def greedy_action(Q, s):
    return int(np.argmax(Q[s]))

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
def train_sarsa(env, episodes=300, alpha=0.1, gamma=1,
                eps_start=0.2, eps_end=0.01, eps_decay=0.995, seed=0):
    """
    Returns:
      Q: defaultdict[state_tuple] -> np.array(nA)
      logs: list of dicts with per-episode metrics:
            {'return', 'steps', 'iters', 'solved', 'final_mean_dist', 'eps'}
    """
    np.random.seed(seed)
    nA = len(env.actions)
    Q = defaultdict(lambda: np.zeros(nA, dtype=float))
    logs = []

    with suppress_render(env):  # no pop-ups while learning
        for ep in range(episodes):
            s = env.reset()
            eps = max(eps_end, eps_start * (eps_decay ** ep))
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
                    target = r + gamma * Q[s2][a2]

                Q[s][a] += alpha * (target - Q[s][a])
                s, a = s2, (a2 if not done else a)

            # episode summary
            final_mean_dist = env.rectangle.get_mean_dist(points=env.points)
            solved = bool(np.all(np.array(env.rectangle.points_distance(points=env.points)) < env.tau))
            logs.append({
                'return': G,
                'steps': steps,
                'iters': total_used_iters,
                'solved': solved,
                'final_mean_dist': float(final_mean_dist),
                'eps': eps,
            })
        
    return Q, logs

# ----- Q-learning training -----
def train_q_learning(
    env,
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

                # 2) step in env
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

    return Q, logs

def run_episode(env, Q, max_steps=100, render=True):
    """
    Runs one episode with greedy policy (ε=0).
    Returns dict: {'return','steps','iters','solved','final_mean_dist'}
    """
    # choose whether to suppress rendering
    class _Null:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    ctx = _Null() if render else suppress_render(env)

    with ctx:
        s = env.reset()
        env.render()
        total = 0.0
        total_used_iters = 0
        steps = 0
        for _ in range(max_steps):
            a = greedy_action(Q, s)
            s, r, done = env.step(a)
            total += r
            steps += 1
            total_used_iters += int(env.latest_used_iters or 0)
            if done:
                break

    final_mean_dist = env.rectangle.get_mean_dist(points=env.points)
    solved = bool(np.all(np.array(env.rectangle.points_distance(points=env.points)) < env.tau))
    return {
        'return': total,
        'steps': steps,
        'iters': total_used_iters,
        'solved': solved,
        'final_mean_dist': float(final_mean_dist),
    }


