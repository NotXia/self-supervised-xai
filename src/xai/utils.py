import math
import torch
from models import ImageClassificationModel, TextClassificationModel, MMClassificationModel, AudioClassificationModel



def get_model_wrapper(model):
    if isinstance(model, ImageClassificationModel): return OnlyImageClassificationModel(model)
    elif isinstance(model, TextClassificationModel): return OnlyTextClassificationModel(model)
    elif isinstance(model, MMClassificationModel): return OnlyTextImageClassificationModel(model)
    elif isinstance(model, AudioClassificationModel): return OnlyAudioClassificationModel(model)


class OnlyImageClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.image_encoder = model.image_encoder
        self.classifier = model.classifier

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


class OnlyTextClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.text_encoder = self.model.text_encoder
        self.classifier = self.model.classifier

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


class OnlyTextImageClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.text_encoder = model.text_encoder
        self.image_encoder = model.image_encoder
        self.fusion = model.fusion
        self.classifier = model.classifier

    def __apply_attribution(self, inputs, attribution):
        return inputs * attribution

    def forward(self, inputs_image, inputs_text, attn_masks, attribution=None):
        if inputs_text.dtype is torch.int64: # Input is token ids
            inputs_text = self.text_encoder.embed(inputs_text)

        if attribution is not None:
            inputs_image = self.__apply_attribution(inputs_image, attribution[0])
            inputs_text = self.__apply_attribution(inputs_text, attribution[1])

        embeds_image, _ = self.image_encoder({"pixel_values": inputs_image})
        embeds_text = self.text_encoder._model(inputs_text, attn_masks)
        
        fusion_embeds = self.fusion(embeds_text, embeds_image, attn_masks).mean(dim=1)
        logits = self.classifier(fusion_embeds)
        
        return logits


class OnlyAudioClassificationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.audio_encoder = model.audio_encoder
        self.classifier = model.classifier

    def __apply_attribution(self, inputs, attribution):
        return inputs * attribution

    def forward(self, inputs, attribution=None):
        if attribution is not None:
            inputs = self.__apply_attribution(inputs, attribution)

        embeds = self.audio_encoder(
            {"input_values": inputs}
        )
        logits = self.classifier(embeds)
        return logits