import math
import torch
from models import ImageClassificationModel, TextClassificationModel



def get_model_wrapper(model):
    if isinstance(model, ImageClassificationModel): return OnlyImageClassificationModel(model)
    elif isinstance(model, TextClassificationModel): return OnlyTextClassificationModel(model)


class OnlyImageClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.image_encoder = model.image_encoder
        self.classifier = model.classifier
        self.lig_layer = self.model.image_encoder.encoder.embeddings

    def __apply_attribution(self, inputs, attribution):
        return inputs * attribution

    def forward(self, inputs, attribution=None):
        if attribution is not None:
            inputs = self.__apply_attribution(inputs, attribution)

        embeds, _ = self.image_encoder(
            {"pixel_values": inputs}
        )
        logits = self.classifier(embeds)
        return logits

    def lig_postprocessing(self, attribution):
        hw = int( math.sqrt(attribution.shape[1]-1) )
        attribution = attribution[:, 1:].mean(dim=2).reshape(-1, 1, hw, hw)
        return (attribution, )


class OnlyTextClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.text_encoder = self.model.text_encoder
        self.classifier = self.model.classifier
        self.lig_layer = self.model.text_encoder._model.encoder.embeddings

    def __apply_attribution(self, static_embeds, attribution):
        return static_embeds * attribution

    def forward(self, static_embeds, attn_masks, attribution=None):
        if static_embeds.dtype is torch.int64: # Input is token ids
            static_embeds = self.text_encoder.embed(static_embeds)

        if attribution is not None:
            static_embeds = self.__apply_attribution(static_embeds, attribution)

        embeds = self.text_encoder._model(static_embeds, attn_masks)
        logits = self.classifier(embeds[:, 0, :])
        return logits

    def lig_postprocessing(self, attribution):
        return (attribution, )