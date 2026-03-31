import torch
from captum.attr import IntegratedGradients
from .utils import *



class IntegratedGradientsAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        self.ig = IntegratedGradients(self.model)
        self.baselines = baselines

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )

        attribution = self.ig.attribute(
            inputs, 
            baselines = self.baselines, 
            target = target, 
            additional_forward_args = additional_forward_args
        )
        return attribution
