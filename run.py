"""
Minimal CLI entrypoint to train/evaluate the rectangle alignment agent outside the notebook.

Defaults mirror `testrun.ipynb`: 1 ray, small MLP, linear eps/alpha decay, random env
generator with offsets/rotations. Stats are printed to stdout; plots are saved by the
environment render when evaluation runs the last episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.agent_ann_base import QNetwork
from src.models.agent_ann_nstep import NStepAgentANN
from src.models.trainer import Trainer
from src.utils.examples_gen import make_env_generator
from src.utils.scheduler import LinearDecayRule, ScheduleParam


def build_env_generator(args):
    rect_params = {
        "center": tuple(args.center),
        "width": args.width,
        "height": args.height,
        "theta": args.theta,
    }
    optimizer_params = {"loss": "soft_l1", "bounds": (-5, 5)}
    rays_params = {"n_rays": args.n_rays}
    rng = np.random.default_rng(args.seed)

    x_lim = tuple(args.x_lim) if args.x_lim else None
    y_lim = tuple(args.y_lim) if args.y_lim else None

    return make_env_generator(
        rect_params=rect_params,
        optimizer_params=optimizer_params,
        rays_params=rays_params,
        offset_range=(
            (args.offset_min_x, args.offset_max_x),
            (args.offset_min_y, args.offset_max_y),
        ),
        rotation_range=(args.rot_min, args.rot_max),
        rng=rng,
        x_lim=x_lim,
        y_lim=y_lim,
    )


def build_agent(args, device):
    state_dim = 4 * args.n_rays
    n_actions = 9  # matches Environment.actions ordering
    network = QNetwork(state_dim, n_actions, hidden_dim=args.hidden_dim)

    eps = ScheduleParam(1.0, LinearDecayRule(start=0.4, end=0.05, steps=args.eps_steps))
    alpha = ScheduleParam(
        1.0, LinearDecayRule(start=0.2, end=0.05, steps=args.alpha_steps)
    )

    agent = NStepAgentANN(
        nA=n_actions,
        alpha=alpha,
        eps=eps,
        network=network,
        device=device,
        n_step=args.n_step,
        debug=args.debug,
    )
    return agent


def save_model(agent, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.Q_net.state_dict(), path)
    print(f"Saved model to {path}")


def maybe_load(agent, path: Path | None, map_location):
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    agent.Q_net.load_state_dict(state)
    agent.Q_target.load_state_dict(state)
    print(f"Loaded checkpoint from {path}")


def run_training(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    env_gen = build_env_generator(args)
    agent = build_agent(args, device)
    trainer = Trainer(envs=env_gen, agent=agent, reuse_per_env=args.reuse_per_env)

    maybe_load(agent, Path(args.checkpoint) if args.checkpoint else None, device)

    print(f"Starting training for {args.episodes} episodes...")
    trainer.fit(episodes=args.episodes, log_interval=args.log_interval)

    if args.save_path:
        save_model(agent, Path(args.save_path))

    if args.eval_after > 0:
        env = env_gen()
        env.reset()
        metrics = trainer.evaluate(
            env, episodes=args.eval_after, show_last_run=not args.no_render
        )
        env.show_gif(filename=args.gif_name)
        print("Eval metrics (last run rendered):")
        for m in metrics:
            print(m)


def run_eval_only(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    env_gen = build_env_generator(args)
    agent = build_agent(args, device)
    trainer = Trainer(envs=env_gen, agent=agent, reuse_per_env=args.reuse_per_env)

    checkpoint = args.checkpoint or f"models/ray{args.n_rays}step100.pt"
    maybe_load(agent, Path(checkpoint), device)

    env = env_gen()
    env.reset()
    metrics = trainer.evaluate(
        env, episodes=args.eval_after or 1, show_last_run=not args.no_render
    )
    env.show_gif(filename=args.gif_name)
    print("Eval metrics:")
    for m in metrics:
        print(m)


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate rectangle agent.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Run training (default).")
    mode.add_argument("--eval-only", action="store_true", help="Skip training.")

    p.add_argument("--episodes", type=int, default=2000, help="Training episodes.")
    p.add_argument("--eval-after", type=int, default=3, help="Eval episodes after train.")
    p.add_argument("--log-interval", type=int, default=10, help="Trainer log interval.")
    p.add_argument("--reuse-per-env", type=int, default=1, help="Reuse env across episodes.")

    # model/env params
    p.add_argument("--n-rays", type=int, default=2, help="Number of rays.")
    p.add_argument("--hidden-dim", type=int, default=32, help="MLP hidden size.")
    p.add_argument("--n-step", type=int, default=1, help="n-step return for agent.")
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--center", nargs=2, type=float, default=(0.0, 0.0))
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--offset-min-x", type=float, default=-5.0)
    p.add_argument("--offset-max-x", type=float, default=5.0)
    p.add_argument("--offset-min-y", type=float, default=-5.0)
    p.add_argument("--offset-max-y", type=float, default=5.0)
    p.add_argument("--rot-min", type=float, default=-np.pi / 2)
    p.add_argument("--rot-max", type=float, default=np.pi / 2)
    p.add_argument("--x-lim", nargs=2, type=float, help="Override x limits.")
    p.add_argument("--y-lim", nargs=2, type=float, help="Override y limits.")
    p.add_argument("--seed", type=int, default=1234)

    # schedules/optimizer
    p.add_argument("--eps-steps", type=int, default=1000, help="ε decay steps.")
    p.add_argument("--alpha-steps", type=int, default=2000, help="LR decay steps.")

    # IO
    p.add_argument("--checkpoint", type=str, help="Path to load weights.")
    p.add_argument("--save-path", type=str, help="Where to save trained weights.")
    p.add_argument("--device", type=str, help="force device, e.g. cpu or cuda:0")
    p.add_argument("--no-render", action="store_true", help="Skip rendering last eval run.")
    p.add_argument("--debug", action="store_true", help="Enable agent debug prints.")
    p.add_argument("--gif-name", type=str, default="last_run.gif", help="Filename for saved GIF.")

    return p.parse_args()


def main():
    args = parse_args()
    if args.eval_only:
        run_eval_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
