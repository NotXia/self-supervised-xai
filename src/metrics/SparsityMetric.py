import torch

from .BaseMetric import BaseMetric


class SparsityMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def accumulate(self, attribution):
        if not isinstance(attribution, tuple):
            attribution = (attribution, )
        
        sparsity = []
        for attr in attribution:
            attr = torch.abs(attr)
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-16)
            sparsity.append( (attr.max() / (attr.mean() + 1e-16)).item() )
        self.values.append( sum(sparsity) / len(sparsity) )