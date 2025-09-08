<!-- Toy example from internship -->

<!-- Tabular methods on rectangle aligning problem -->

# Rectangle Alignment

This topy example implements a reinforcement learning environment where an agent learns to align a rectangle with a set of target points.
The task is framed as a discrete state–action problem suitable for Q-learning, SARSA, and n-step variants.

## State and Action Space
- **States**: Defined by
  1. Proximity counts of points to rectangle sides.
  2. Distance distribution categories.
- **Actions**: Optimization steps, direct moves, and rotations.
- **Total space size**: ~3,000,000 state–action pairs.

## Agent
- Uses Q-learning / SARSA / n-step variants to update the Q-table.
- **Exploration methods**:
  - ε-greedy with decay.
  - Upper Confidence Bound (UCB) with exploration bonus.
- **Fall-back logic**:
  - Rotation forced if updates are too small (to escape plateaus).
  - Escape from repeated poses to avoid loops.

## Reward Function
- Negative reward proportional to optimization cost.
- Positive bonuses for full alignment.
- Penalties for going out of bounds.
- Costs are scaled down for exploratory moves and rotations to promote escaping local optima.


## Results
[Click here to view the GIF](https://imgur.com/S14EcEU)

[Click here to view the GIF](https://imgur.com/wu4Msu2)
