import torch
from captum.attr import DeepLiftShap, GradientShap
from .utils import *



class DeepLiftShapAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        self.deeplift = DeepLiftShap(self.model)
        self.baselines = baselines

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )
        return self.deeplift.attribute(
            inputs, 
            baselines = self.baselines, 
            target = target,
            additional_forward_args = additional_forward_args
        )


class GradientShapAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        self.gshap = GradientShap(self.model)
        self.baselines = baselines

    def __call__(self, inputs, target, n_samples=5, stdevs=0.0, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )
        return self.gshap.attribute(
            inputs, 
            baselines = self.baselines, 
            target = target,
            n_samples = n_samples,
            stdevs = stdevs,
            additional_forward_args = additional_forward_args
        )