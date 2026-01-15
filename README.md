# Rectangle Alignment (RL Toy)

## Quick Start
```
git clone https://gitlab.ado.lan.watchout.work/labodata/reinforcement-learning
cd reinforcement-learning
git checkout toyExample-SCh2

```

Run training/eval without the notebook.

- Train + short eval (defaults mirror notebook: 1 ray, 2000 eps, eval 3 runs):
  ```
  python run.py --episodes 2000 --eval-episodes 3
  ```

- Eval only from a checkpoint:
  ```
  python run.py --eval-only --checkpoint models/ray1step100.pt --eval-episodes 1
  ```


## Overview
A small reinforcement learning environment where an agent aligns a movable rectangle with a target polygon (same size, displaced). The agent reads continuous ray-based signals and chooses discrete moves/rotations to minimize misalignment.

## Environment
- **State**: Continuous vector of signed distances and directions from `n` rays
  `s = [d₁..dₙ, cosθ₁, sinθ₁, …, cosθₙ, sinθₙ]` (length `4n`)
- **Actions**: Translations along local axes, rotations, and small optimizer calls
- **Reward**: Normalized distance term + improvement term − step cost + bonus for **all-touching** − out-of-bounds penalty
  (rotation cost slightly increased to discourage meaningless spinning)



## Testing
All agents are evaluated on the **same** set of randomly generated environments. Metrics are reported as mean ± std over 100 episodes.

| n_rays| Reward            | Steps           | Optimizer iters    |
|:-----:|:------------------|:----------------|:-------------------|
| 1     | −8.16 ± 16.06     | 10.11 ± 11.42   | 19.56 ± 25.89      |
| 2     | −3.66 ± 8.13      | 8.93 ± 6.49     | 5.11 ± 6.34        |
| 3     | −5.83 ± 9.74      | 10.86 ± 7.90    | 6.60 ± 8.63        |


## Examples
- 1 ray: https://imgur.com/uIDSc4H
- 2 rays: https://imgur.com/6B5xWwX
- 3 rays: https://imgur.com/S0A9d4Z

## CLI entrypoint (`run.py`)


Notable flags:
- `--n-rays` (default 2): sets ray count and state dim (`4 * n_rays`).
- `--hidden-dim` (default 32): MLP width; set 0/neg for linear.
- `--n-step` (default 1): n-step return length for the ANN agent.
- `--offset-min-* / --offset-max-*`, `--rot-min / --rot-max`: randomness for env generator.
- `--eps-steps`, `--alpha-steps`: linear decay horizons for ε and LR schedules.
- `--checkpoint`: load weights; `--save-path`: write trained weights.
- `--no-render`: skip rendering during the final eval episode.
- `--gif-name`: filename for saved GIF.

Outputs:
- Model weights at `--save-path` (if provided).
- Frames/plots from the last eval episode (unless `--no-render` is set).
