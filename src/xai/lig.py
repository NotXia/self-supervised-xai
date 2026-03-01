import torch
from captum.attr import LayerIntegratedGradients
from .utils import *



class LayerIntegratedGradientsAttribution():
    def __init__(self, model, baselines):
        self.model = get_model_wrapper(model)
        if isinstance(self.model.lig_layer, tuple):
            self.lig = [ LayerIntegratedGradients(self.model, l) for l in self.model.lig_layer ]
        else:
            self.lig = LayerIntegratedGradients(self.model, self.model.lig_layer)
        self.baselines = baselines

    def __call__(self, inputs, target, additional_forward_args=None):
        if type(inputs) is not tuple:
            inputs = (inputs, )

        attribution = None
        if isinstance(self.lig, list):
            attribution = []
            for lig in self.lig:
                attribution.append(
                    lig.attribute(
                        inputs, 
                        baselines = self.baselines, 
                        target = target, 
                        additional_forward_args = additional_forward_args
                    )
                )
        else:
            attribution = self.lig.attribute(
                inputs, 
                baselines = self.baselines, 
                target = target, 
                additional_forward_args = additional_forward_args
            )
        return self.model.lig_postprocessing(attribution)
