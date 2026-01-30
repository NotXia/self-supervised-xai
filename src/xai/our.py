import torch
from models import TextClassificationModel, ImageClassificationModel


class OurAttribution():
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, inputs):
        if isinstance(self.model, TextClassificationModel):
            _, attr = self.model({
                "input_ids": inputs[0],
                "attention_mask": inputs[1]
            })
        elif isinstance(self.model, ImageClassificationModel):
            _, attr = self.model({
                "pixel_values": inputs[0],
            })
        attr = attr.cpu()
        return (attr, )
