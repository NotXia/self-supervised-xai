import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from sklearn.metrics import auc

from .BaseMetric import BaseMetric
from xai import OnlyImageClassificationModel, OnlyTextClassificationModel, OnlyTextImageClassificationModel


class DeletionAUCMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device


    def accumulate(self, model, input, attribution, steps=10):
        if isinstance(model, OnlyImageClassificationModel):
            self.values.append( self.__auc_image_classifier(model, input, attribution, steps) )
        elif isinstance(model, OnlyTextClassificationModel):
            self.values.append( self.__auc_text_classifier(model, input, attribution, steps) )
        elif isinstance(model, OnlyTextImageClassificationModel):
            self.values.append( self.__auc_text_image_classifier(model, input, attribution, steps) )
        else:
            raise NotImplementedError()


    @torch.no_grad()
    def __auc_image_classifier(self, model, inputs, attribution, steps):
        inputs = inputs[0]
        attribution = torchvision.transforms.functional.resize(attribution, (inputs.shape[2], inputs.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-16)
        attribution = attribution.to(self.device)
        points = []

        for threshold in np.linspace(0.0, 1.0, steps):
            logits = model(inputs, attribution=None)
            logits_masked = model(inputs, attribution=(attribution > threshold))

            logits = F.softmax(logits, dim=1)
            logits_masked = F.softmax(logits_masked, dim=1)
            pred = torch.argmax(logits, dim=1)[0].item()
            
            average_drop = max((logits[0, pred] - logits_masked[0, pred]).item(), 0)
            points.append( (threshold, average_drop) )

        return auc([p[0]for p in points], [p[1]for p in points])


    @torch.no_grad()
    def __auc_text_classifier(self, model, inputs, attribution, steps):
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-16)
        attribution = attribution.to(self.device)
        input_ids, attn_masks = inputs
        points = []

        for threshold in np.linspace(0.0, 1.0, steps):
            logits = model(input_ids, attn_masks, attribution=None)
            logits_masked = model(input_ids, attn_masks, attribution=(attribution > threshold))
            pred = torch.argmax(logits, dim=1)[0].item()

            average_drop = max((logits[0, pred] - logits_masked[0, pred]).item(), 0)
            points.append( (threshold, average_drop) )

        return auc([p[0]for p in points], [p[1]for p in points])


    @torch.no_grad()
    def __auc_text_image_classifier(self, model, inputs, attribution, steps):
        inputs_image, inputs_text, attn_masks = inputs
        attribution = list(attribution)
        attribution[0] = torchvision.transforms.functional.resize(attribution[0], (inputs_image.shape[2], inputs_image.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        attribution[0] = ( (attribution[0] - attribution[0].min()) / (attribution[0].max() - attribution[0].min() + 1e-16) ).to(self.device)
        attribution[1] = ( (attribution[1] - attribution[1].min()) / (attribution[1].max() - attribution[1].min() + 1e-16) ).to(self.device)
        points = []

        for threshold in np.linspace(0.0, 1.0, steps):
            logits = model(inputs_image, inputs_text, attn_masks, attribution=None)
            logits_masked = model(inputs_image, inputs_text, attn_masks, attribution=[a > threshold for a in attribution])
            pred = torch.argmax(logits, dim=1)[0].item()

            average_drop = max((logits[0, pred] - logits_masked[0, pred]).item(), 0)
            points.append( (threshold, average_drop) )

        return auc([p[0]for p in points], [p[1]for p in points])
