# RL Exercise for Rectangle Alignment

This repository implements a continuous-state, continuous-action reinforcement learning (RL) environment, where an agent learns to align a movable rectangle with a displaced target polygon through translations and rotations.
The agent is trained using **Proximal Policy Optimization (PPO)** within the `Gymnasium` and `Stable-Baselines3` frameworks.

---

## Overview
<!--
### PPO with Continuous Actions -->
The agent acts in a continuous 3D action space \( a = $[\Delta x, \Delta y, \Delta \theta]$ \), where each component represents local translations and rotations of the rectangle.
Actions are bounded and scaled by maximum step sizes \($[s_x, s_y, s_\theta]$ = [1.2, 1.2, $\pi/4$]\), ensuring consistent motion magnitudes within the bounded region \($[-10, 10]^2$\).

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