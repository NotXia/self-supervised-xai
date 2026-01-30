import torch
import torch.nn.functional as F
import torchvision

from .BaseMetric import BaseMetric
from xai import OnlyImageClassificationModel, OnlyTextClassificationModel


class AverageDropMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device


    def accumulate(self, model, input, attribution):
        if isinstance(model, OnlyImageClassificationModel):
            self.values.append( self.__ad_image_classifier(model, input, attribution) )
        elif isinstance(model, OnlyTextClassificationModel):
            self.values.append( self.__ad_text_classifier(model, input, attribution) )
        else:
            raise NotImplementedError()


    @torch.no_grad()
    def __ad_image_classifier(self, model, inputs, attribution):
        inputs = inputs[0]
        attribution = torchvision.transforms.functional.resize(attribution, (inputs.shape[2], inputs.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())

        logits = model(inputs, attribution=None)
        logits_masked = model(inputs, attribution=attribution.to(self.device))

        logits = F.softmax(logits, dim=1)
        logits_masked = F.softmax(logits_masked, dim=1)
        pred = torch.argmax(logits, dim=1)[0].item()

        return max((logits[0, pred] - logits_masked[0, pred]).item(), 0)


    @torch.no_grad()
    def __ad_text_classifier(self, model, inputs, attribution):
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        input_ids, attn_masks = inputs

        logits = model(input_ids, attn_masks, attribution=None)
        logits_masked = model(input_ids, attn_masks, attribution=attribution.to(self.device))
        pred = torch.argmax(logits, dim=1)[0].item()

        return max((logits[0, pred] - logits_masked[0, pred]).item(), 0)
