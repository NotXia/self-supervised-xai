import torch
from models import TextClassificationModel, ImageClassificationModel, MMClassificationModel, AudioClassificationModel


class OurAttribution():
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, inputs, label=None):
        if isinstance(self.model, TextClassificationModel):
            _, attr = self.model({
                "input_ids": inputs[0],
                "attention_mask": inputs[1]
            })
            attr = attr.cpu()
            return (attr, )
        elif isinstance(self.model, ImageClassificationModel):
            _, attr = self.model({
                "pixel_values": inputs[0],
            })
            attr = attr.cpu()
            return (attr, )
        elif isinstance(self.model, MMClassificationModel):
            logits, text_weights, image_weights = self.model({
                "pixel_values": inputs[0],
            },
            {
                "input_ids": inputs[1],
                "attention_mask": inputs[2]
            })
            text_weights, image_weights = text_weights.cpu(), image_weights.cpu()
            return (image_weights, text_weights)
        elif isinstance(self.model, AudioClassificationModel):
            _, attr = self.model({
                "input_values": inputs[0],
            })
            attr = attr.cpu()
            return (attr, )
