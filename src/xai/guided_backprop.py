import torch
from captum.attr import GuidedBackprop
from .utils import *


class GuidedBackpropAttribution():
    def __init__(self, model):
        self.model = get_model_wrapper(model)
        self.gb = GuidedBackprop(self.model)

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )
        return self.gb.attribute(
            inputs, 
            target = target,
            additional_forward_args = additional_forward_args
        )