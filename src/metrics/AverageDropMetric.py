import torch
import torch.nn.functional as F
import torchvision

from .BaseMetric import BaseMetric
from xai import OnlyImageClassificationModel, OnlyTextClassificationModel, OnlyTextImageClassificationModel, OnlyAudioClassificationModel


class AverageDropMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device


    def accumulate(self, model, input, attribution):
        if isinstance(model, OnlyImageClassificationModel):
            self.values.append( self.__ad_image_classifier(model, input, attribution) )
        elif isinstance(model, OnlyTextClassificationModel):
            self.values.append( self.__ad_text_classifier(model, input, attribution) )
        elif isinstance(model, OnlyTextImageClassificationModel):
            self.values.append( self.__ad_text_image_classifier(model, input, attribution) )
        elif isinstance(model, OnlyAudioClassificationModel):
            self.values.append( self.__ad_audio_classifier(model, input, attribution) )
        else:
            raise NotImplementedError()


    @torch.no_grad()
    def __ad_image_classifier(self, model, inputs, attribution):
        inputs = inputs[0]
        attribution = torchvision.transforms.functional.resize(attribution, (inputs.shape[2], inputs.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        attribution = torch.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-16)

        logits = model(inputs, attribution=None)
        logits_masked = model(inputs, attribution=attribution.to(self.device))

        logits = F.softmax(logits, dim=1)
        logits_masked = F.softmax(logits_masked, dim=1)
        pred = torch.argmax(logits, dim=1)[0].item()

        return max(((logits[0, pred] - logits_masked[0, pred])/logits[0, pred]).item(), 0)


    @torch.no_grad()
    def __ad_text_classifier(self, model, inputs, attribution):
        attribution = torch.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-16)
        input_ids, attn_masks = inputs

        logits = model(input_ids, attn_masks, attribution=None)
        logits_masked = model(input_ids, attn_masks, attribution=attribution.to(self.device))

        logits = F.softmax(logits, dim=1)
        logits_masked = F.softmax(logits_masked, dim=1)
        pred = torch.argmax(logits, dim=1)[0].item()

        return max(((logits[0, pred] - logits_masked[0, pred])/logits[0, pred]).item(), 0)


    @torch.no_grad()
    def __ad_text_image_classifier(self, model, inputs, attribution):
        inputs_image, inputs_text, attn_masks = inputs
        attribution = list(attribution)
        attribution[0] = torchvision.transforms.functional.resize(attribution[0], (inputs_image.shape[2], inputs_image.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        attribution[0] = torch.abs(attribution[0])
        attribution[1] = torch.abs(attribution[1])
        attribution[0] = ( (attribution[0] - attribution[0].min()) / (attribution[0].max() - attribution[0].min() + 1e-16) ).to(self.device)
        attribution[1] = ( (attribution[1] - attribution[1].min()) / (attribution[1].max() - attribution[1].min() + 1e-16) ).to(self.device)

        logits = model(inputs_image, inputs_text, attn_masks, attribution=None)
        logits_masked = model(inputs_image, inputs_text, attn_masks, attribution=attribution)

        logits = F.softmax(logits, dim=1)
        logits_masked = F.softmax(logits_masked, dim=1)
        pred = torch.argmax(logits, dim=1)[0].item()

        return max(((logits[0, pred] - logits_masked[0, pred])/logits[0, pred]).item(), 0)


    @torch.no_grad()
    def __ad_audio_classifier(self, model, inputs, attribution):
        inputs = inputs[0]
        attribution = torch.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-16)

        logits = model(inputs, attribution=None)
        logits_masked = model(inputs, attribution=attribution.to(self.device))

        logits = F.softmax(logits, dim=1)
        logits_masked = F.softmax(logits_masked, dim=1)
        pred = torch.argmax(logits, dim=1)[0].item()

        return max(((logits[0, pred] - logits_masked[0, pred])/logits[0, pred]).item(), 0)