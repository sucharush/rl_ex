#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_RECT_PARAMS = dict(center=(0, 0), width=4.0, height=2.0, theta=np.pi / 6)
DEFAULT_OPTIMIZER_PARAMS = dict(loss="soft_l1", bounds=(-5, 5))
DEFAULT_SEGS_PARAMS = dict(
    num_points=10,
    jitter_n=0.05,
    jitter_t=0.1,
    offset_range=((-5, 5), (-5, 5)),
    rotation_range=(-np.pi / 2, np.pi / 2),
    seed=42,
)
DEFAULT_RAYS_PARAMS = dict(n_rays=3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run/visualize GymRectangleEnv with SB3 models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model.")
    eval_parser.add_argument("--model-path", default="models/improved3/model.zip")
    eval_parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--episodes", type=int, default=1)
    eval_parser.add_argument("--max-steps", type=int, default=100)
    eval_parser.add_argument(
        "--render",
        choices=["human", "rgb", "none"],
        default="human",
        help="human: interactive window, rgb: capture frames, none: no render.",
    )
    eval_parser.add_argument("--video-path", default="")
    eval_parser.add_argument("--fps", type=int, default=3)
    eval_parser.add_argument("--render-sec", type=float, default=0.3)
    eval_parser.add_argument("--seed", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Train a new model.")
    train_parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--total-timesteps", type=int, default=30000)
    train_parser.add_argument("--save-path", default="models/rectangle_ppo.zip")
    train_parser.add_argument("--log-dir", default="runs")
    train_parser.add_argument("--use-wandb", action="store_true")
    train_parser.add_argument("--wandb-project", default="rectangle-alignment")
    train_parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def make_env(render_mode: str, seed: int | None):
    from src.environments.gym_env import GymRectangleEnv

    env = GymRectangleEnv(
        rect_params=DEFAULT_RECT_PARAMS,
        optimizer_params=DEFAULT_OPTIMIZER_PARAMS,
        rays_params=DEFAULT_RAYS_PARAMS,
        segs_params=DEFAULT_SEGS_PARAMS,
        render_mode=render_mode if render_mode != "none" else None,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def load_model(algo: str, model_path: str, env, device: str):
    if algo == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env, device=device)
    if algo == "SAC":
        from stable_baselines3 import SAC

        return SAC.load(model_path, env=env, device=device)
    raise ValueError(f"Unsupported algo: {algo}")


def run_eval(args):
    render_mode = "rgb" if args.video_path else args.render
    env = make_env("rgb_array" if render_mode == "rgb" else render_mode, args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(args.algo, str(model_path), env, args.device)

    all_rewards = []
    all_lengths = []
    frames = []

    for _ in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        episode_reward = 0.0
        steps = 0

        if args.video_path:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if render_mode == "human":
                env.render(sec=args.render_sec)
            elif render_mode == "rgb":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        all_lengths.append(steps)

    env.close()

    if args.video_path and frames:
        import imageio

        os.makedirs(Path(args.video_path).parent, exist_ok=True)
        with imageio.get_writer(args.video_path, fps=args.fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)

    print(
        f"episodes={len(all_rewards)} "
        f"reward_mean={np.mean(all_rewards):.3f} "
        f"reward_std={np.std(all_rewards):.3f} "
        f"len_mean={np.mean(all_lengths):.2f}"
    )


def maybe_init_wandb(use_wandb: bool, project: str, config: dict):
    if not use_wandb:
        return None
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except Exception as exc:  # pragma: no cover - best-effort optional
        raise RuntimeError(
            "wandb requested but not available. Install wandb or omit --use-wandb."
        ) from exc

    wandb.init(project=project, config=config, sync_tensorboard=True)
    callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb.run.id}",
        log="all",
    )
    return wandb, callback


def run_train(args):
    env = make_env("rgb_array", args.seed)

    config = dict(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        env_name="GymRectangleEnv-v0",
    )
    wandb_mod = None
    wandb_callback = None
    if args.use_wandb:
        wandb_mod, wandb_callback = maybe_init_wandb(
            args.use_wandb, args.wandb_project, config
        )

    if args.algo == "PPO":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )
    else:
        from stable_baselines3 import SAC

        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=wandb_callback,
        log_interval=32,
    )
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    model.save(args.save_path)
    env.close()

    if wandb_mod is not None:
        wandb_mod.finish()


def main():
    args = parse_args()
    if args.command == "eval":
        run_eval(args)
    elif args.command == "train":
        run_train(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
