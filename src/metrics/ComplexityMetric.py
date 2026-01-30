import torch
import torch.nn.functional as F

from .BaseMetric import BaseMetric


class ComplexityMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def accumulate(self, attribution):
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        self.values.append( torch.norm(attribution).item() )