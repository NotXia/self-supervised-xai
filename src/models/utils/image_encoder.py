import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel

from .utils import get_vit_config



class ImageEncoderModel(L.LightningModule):
    def __init__(self, model_card):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_card)
        self.encoder = AutoModel.from_pretrained(model_card)
        encoder_config = get_vit_config(model_card)
        self.hidden_size = encoder_config["hidden_size"]
        self.in_resolution = encoder_config["in_resolution"]
        self.out_resolution = encoder_config["out_resolution"]
        
    def preprocess(self, images):
        return self.processor(
            images = images,
            return_tensors = "pt",
            do_rescale = False,
        ).to(self.device)

    def forward(self, inputs):
        encoder_out = self.encoder(**inputs, output_hidden_states=True)
        return encoder_out.last_hidden_state, encoder_out.hidden_states


    def embed_with_masker(self, inputs, masker_model, conditioning=None):
        embeds, hidden_states = self.forward(inputs)

        weights = masker_model(hidden_states, conditioning)

        inputs_to_use = inputs.copy()
        inputs_to_use["pixel_values"] = (inputs_to_use["pixel_values"] * weights)
        masked_embeds, _ = self.forward(inputs_to_use)

        return masked_embeds, weights