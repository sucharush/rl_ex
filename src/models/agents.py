from src.models.agent_base import AgentBase
import numpy as np
class QLearningAgent(AgentBase):

    def fallback_action(self):
        return self.nA - 1

    def step(self, s, a, r, s2, done, stalled=False):
        if stalled:
            a = self.fallback_action(s)
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s][a] += self.alpha() * (target - self.Q[s][a])
        return self.select_action(s2)

class SarsaAgent(AgentBase):

    def fallback_action(self):
        return self.nA - 1
    def step(self, s, a, r, s2, done, stalled=False):
        a2 = self.select_action(s2)
        if stalled:
            a2 = self.fallback_action(s)
        target = r if done else r + self.gamma * self.Q[s2][a2]
        self.Q[s][a] += self.alpha() * (target - self.Q[s][a])
        return a2
