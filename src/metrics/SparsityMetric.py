import torch

from .BaseMetric import BaseMetric


class SparsityMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def accumulate(self, attribution):
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        self.values.append( (attribution.max() / attribution.mean()).item() )