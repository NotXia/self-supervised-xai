import torch
from captum.attr import DeepLift, DeepLiftShap, Saliency, LayerIntegratedGradients
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


class LayerIntegratedGradientsAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        self.lig = LayerIntegratedGradients(self.model, self.model.lig_layer)
        self.baselines = baselines

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )
        attribution = self.lig.attribute(
            inputs, 
            baselines = self.baselines, 
            target = target, 
            additional_forward_args = additional_forward_args
        )
        return self.model.lig_postprocessing(attribution)