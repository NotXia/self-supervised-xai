import torch
from captum.attr import Saliency
from .utils import *



class SaliencyAttribution():
    def __init__(self, model):
        self.model = get_model_wrapper(model)
        self.saliency = Saliency(self.model)

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )
        return self.saliency.attribute(
            inputs, 
            target = target,
            additional_forward_args = additional_forward_args
        )
