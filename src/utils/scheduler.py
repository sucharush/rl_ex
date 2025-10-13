from abc import ABC, abstractmethod

class UpdateRule(ABC):
    @abstractmethod
    def __call__(self, t: int, value: float) -> float:
        """Return the updated parameter value at step t."""
        pass

# update functions
class ConstantRule(UpdateRule):
    def __init__(self, val: float):
        self.val = val
    def __call__(self, t, value):
        return self.val

class LinearDecayRule(UpdateRule):
    def __init__(self, start: float, end: float, steps: int):
        self.start, self.end, self.steps = start, end, steps
    def __call__(self, t, value):
        frac = min(t / self.steps, 1.0)
        return self.start + (self.end - self.start) * frac

class ExponentialDecayRule(UpdateRule):
    def __init__(self, start: float, gamma: float):
        self.start, self.gamma = start, gamma
    def __call__(self, t, value):
        return self.start * (self.gamma ** t)



# Scheduler class

class ScheduleParam:
    def __init__(self, init, update_fn: UpdateRule):
        self.value = init
        self.update_fn = update_fn
        self.t = 0

    def step(self):
        self.t += 1
        self.value = self.update_fn(self.t, self.value)
        return self.value

    def __call__(self):
        return self.value


