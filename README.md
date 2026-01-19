# RL Exercise for Rectangle Alignment

This repository implements a continuous-state, continuous-action reinforcement learning (RL) environment, where an agent learns to align a movable rectangle with a displaced target polygon through translations and rotations.
The agent is trained using **Proximal Policy Optimization (PPO)** within the `Gymnasium` and `Stable-Baselines3` frameworks.

---

## Overview
<!--
### PPO with Continuous Actions -->
The agent acts in a continuous 3D action space \( a = $[\Delta x, \Delta y, \Delta \theta]$ \), where each component represents local translations and rotations of the rectangle.
Actions are bounded and scaled by maximum step sizes \($[s_x, s_y, s_\theta]$ = [1.2, 1.2, $\pi/4$]\), ensuring consistent motion magnitudes within the bounded region \($[-10, 10]^2$\).

## How-to
### Train
```bash
python3 run_gym.py train --total-timesteps 200 --save-path models/rectangle_ppo_test.zip
```
### Evaluate (load a model)
```bash
python3 run_gym.py eval --model-path models/improved3/model.zip --algo PPO --episodes 1 --max-steps 50 --render human
```

### Save a video
```bash
python3 run_gym.py eval --render rgb --video-path videos/eval_test.mp4 --episodes 1 --max-steps 5 --fps 3
```

### Train with wandb
```bash
python3 run_gym.py train --use-wandb --wandb-project rectangle-alignment --total-timesteps 200 --save-path models/rectangle_ppo_test.zip
```



---

- **Frameworks**: `Gymnasium`, `Stable-Baselines3`
- **Curriculum stages**: progressively refined thresholds and reward shaping across four training phases
---

##  Results
| Stage | Description | Avg Reward | Avg Episode Length |
|--------|--------------|-------------|--------------------|
| 1 | Baseline  | −0.05 ± 6.41 | 9.66 ± 10.44 |
| 2 | Fine-tune I | 2.85 ± 3.65 | 5.86 ± 4.62 |
| 3 | Fine-tune II | 3.90 ± 1.32 | 4.98 ± 2.97 |
| 4 | Fine-tune III | **4.08 ± 0.75** | **4.46 ± 2.34** |


PPO exhibits steady improvement in both reward and efficiency, confirming the effectiveness of the multi-stage curriculum.


## Examples
- [vedio](https://imgur.com/AgMvvN7)
- [vedio](https://imgur.com/BVKMoo5)



## Parameters of `run_gym.py`
- `--render`: `human` opens an interactive window, `rgb` captures frames for a video, `none` disables rendering.
- `--video-path`: write an `.mp4` when `--render rgb` is used.
- `--fps`: video frames per second for `--video-path`.
- `--episodes`: number of evaluation episodes to run.
- `--max-steps`: cap on steps per episode during evaluation.
- `--algo`: which SB3 algorithm to load/train (`PPO` or `SAC`).
- `--device`: SB3 device string (e.g., `cpu`, `cuda`).
- `--total-timesteps`: training budget.
- `--save-path`: output path for the trained model.
- `--log-dir`: tensorboard log directory for training.
- `--use-wandb`: enable wandb logging (optional).
- `--wandb-project`: wandb project name when `--use-wandb` is set.
- `--seed`: RNG seed for reproducible resets.
