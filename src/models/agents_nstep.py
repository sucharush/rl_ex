from models.agent_base_nstep import NStepAgentBase
import numpy as np

class NStepQLearningAgent(NStepAgentBase):
    def compute_target(self, s, a):
        return np.max(self.Q[s])
    
class NStepSarsaAgent(NStepAgentBase):
    def compute_target(self, s, a):
        return self.Q[s][a]