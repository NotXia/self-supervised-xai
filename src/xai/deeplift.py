import torch
from captum.attr import DeepLift
from .utils import *


class DeepLiftAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        self.deeplift = DeepLift(self.model)
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