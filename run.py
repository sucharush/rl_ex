#!/usr/bin/env python3
"""
Entrypoint to train and evaluate rectangle-alignment agents.

Usage example:
    python3 run.py --agent nstep_sarsa --episodes 1000 --gif-path logs/run.gif
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.models.agents import QLearningAgent, SarsaAgent
from src.models.agents_nstep import NStepQLearningAgent, NStepSarsaAgent
from src.models.trainer import Trainer
from src.utils.examples_gen import make_env_generator
from src.utils.scheduler import ExponentialDecayRule, LinearDecayRule, ScheduleParam


DEFAULT_RECT_PARAMS = dict(center=(0, 0), width=4.0, height=2.0, theta=np.pi / 6)
DEFAULT_OPTIMIZER_PARAMS = dict(loss="soft_l1", bounds=(-5, 5))
DEFAULT_POINTS_PARAMS = dict(
    num_points=20,
    offset=(1, 1),
    rotation=np.pi / 6,
    jitter_t=0.05,
    jitter_n=0.01,
)
DEFAULT_OFFSET_RANGE = ((-10, 10), (-10, 10))
DEFAULT_ROTATION_RANGE = (-np.pi / 2, np.pi / 2)
DEFAULT_XLIM = (-15, 15)
DEFAULT_YLIM = (-15, 15)


def make_schedules() -> Tuple[ScheduleParam, ScheduleParam, ScheduleParam]:
    """Return fresh schedulers so multiple agents do not share state."""
    eps = ScheduleParam(1.0, LinearDecayRule(start=0.4, end=0.001, steps=1000))
    alpha = ScheduleParam(1.0, LinearDecayRule(start=0.2, end=0.05, steps=1000))
    c = ScheduleParam(1.0, ExponentialDecayRule(start=1.0, gamma=0.999))
    return alpha, eps, c


def build_env_generator(seed: int):
    rng = np.random.default_rng(seed)
    points_params = dict(DEFAULT_POINTS_PARAMS)
    points_params["rng"] = np.random.default_rng(seed)
    env_generator = make_env_generator(
        DEFAULT_RECT_PARAMS,
        DEFAULT_OPTIMIZER_PARAMS,
        points_params,
        offset_range=DEFAULT_OFFSET_RANGE,
        rotation_range=DEFAULT_ROTATION_RANGE,
        x_lim=DEFAULT_XLIM,
        y_lim=DEFAULT_YLIM,
        rng=rng,
    )
    sample_env = env_generator()
    nA = len(sample_env.actions)
    return env_generator, nA


def build_agent(agent_name: str, nA: int, policy: str, n_step: int, gamma: float):
    alpha, eps, c = make_schedules()
    # if agent_name == "q":
    #     return QLearningAgent(nA=nA, alpha=alpha, eps=eps, c=c, gamma=gamma, policy=policy)
    # if agent_name == "sarsa":
    #     return SarsaAgent(nA=nA, alpha=alpha, eps=eps, c=c, gamma=gamma, policy=policy)
    if agent_name == "q":
        return NStepQLearningAgent(
            nA=nA, alpha=alpha, eps=eps, c=c, gamma=gamma, policy=policy, n_step=n_step
        )
    if agent_name == "sarsa":
        return NStepSarsaAgent(
            nA=nA, alpha=alpha, eps=eps, c=c, gamma=gamma, policy=policy, n_step=n_step
        )
    raise ValueError(f"Unknown agent '{agent_name}'")


def run_training(args):
    env_generator, nA = build_env_generator(args.seed)
    agent = build_agent(args.agent, nA, args.policy, args.n_step, args.gamma)
    trainer = Trainer(envs=env_generator, agent=agent, reuse_per_env=args.reuse_per_env)
    trainer.fit(episodes=args.episodes, log_interval=args.log_interval)
    return trainer, env_generator


def run_evaluation(trainer: Trainer, env_generator, eval_episodes: int, gif_path: Path, skip_gif: bool = False):
    env = env_generator()
    metrics = trainer.evaluate(env, episodes=eval_episodes)
    if not skip_gif:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        env.show_gif(str(gif_path))
    return metrics


def save_outputs(trainer: Trainer, metrics, log_path: Path, metrics_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if trainer.logs:
        pd.DataFrame(trainer.logs).to_csv(log_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate rectangle-alignment agents.")
    parser.add_argument("--agent", choices=["q", "sarsa",], default="q")
    parser.add_argument("--policy", choices=["ucb", "egreedy"], default="ucb")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1, dest="n_step")
    parser.add_argument("--reuse-per-env", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif-path", type=Path, default=Path("logs/run.gif"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/training_logs.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/eval_metrics.csv"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a minimal sanity check (episodes=1, eval_episodes=1, reuse=1, log_interval=1).",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF creation during evaluation to save time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.episodes = 1
        args.eval_episodes = 1
        args.reuse_per_env = 1
        args.log_interval = 1
    trainer, env_generator = run_training(args)
    metrics = run_evaluation(trainer, env_generator, args.eval_episodes, args.gif_path, skip_gif=args.skip_gif)
    save_outputs(trainer, metrics, args.log_path, args.metrics_path)
    print(f"Saved training logs to {args.log_path}")
    print(f"Saved evaluation metrics to {args.metrics_path}")
    if not args.skip_gif:
        print(f"Saved evaluation GIF to {args.gif_path}")


if __name__ == "__main__":
    main()
