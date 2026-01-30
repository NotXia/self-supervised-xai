import abc
import numpy as np


class BaseMetric(abc.ABC):
    def __init__(self):
        self.values = []

    def compute(self):
        return np.mean(self.values), np.std(self.values)

    def accumulate(self):
        pass