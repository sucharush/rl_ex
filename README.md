# Reinforcement Learning Toy Projects

This repository hosts three toy RL projects in rectangle alignment.

## Notes
Study notes for reinforcement learning:
see [Project Wiki](https://github.com/sucharush/rl_ex/wiki).

## Quick Start
```bash
git clone https://github.com/sucharush/rl_ex
cd reinforcement-learning
```

## Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Projects Overview
### 1) Tabular RL (Discrete State/Action)
Focus: classic MDPs using tabular methods.
Algorithms: SARSA, Q-learning.
```bash
git checkout toyExample-SCh
```

### 2) Continuous State + Discrete Action (DQN family)
Focus: function approximation for discrete action spaces.
Algorithms: DQN and common extensions (e.g., target networks, replay).
```bash
git checkout toyExample2-SCh
```

### 3) Continuous State + Continuous Action (PPO/SAC)
Focus: policy-gradient and actor-critic methods for continuous control.
Algorithms: PPO and SAC on a custom environment.
```bash
git checkout toyExample3-SCh
```

