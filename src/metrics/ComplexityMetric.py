import torch
import torch.nn.functional as F

from .BaseMetric import BaseMetric


class ComplexityMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def accumulate(self, attribution):
        if not isinstance(attribution, tuple):
            attribution = (attribution, )

        complexity = []
        for attr in attribution:
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-16)
            complexity.append( torch.norm(attr).item() )
        self.values.append( sum(complexity) / len(complexity) )