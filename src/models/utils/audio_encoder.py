import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from transformers import Wav2Vec2FeatureExtractor, AutoModel



class AudioEncoderModel(L.LightningModule):
    def __init__(self, model_card):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_card)
        self.encoder = AutoModel.from_pretrained(model_card)
        self.hidden_size = self.encoder.config.hidden_size
        self.sampling_rate = self.processor.sampling_rate

    def preprocess(self, audio):
        out = self.processor(
            audio,
            sampling_rate = self.sampling_rate, # Data should arrive with the correct sampling rate already
            return_tensors = "pt",
        ).to(self.device)
        out["input_values"] = out["input_values"].squeeze(0)
        return out

    def forward(self, inputs):
        encoder_out = self.encoder(**inputs)
        return encoder_out.last_hidden_state


    def embed_with_masker(self, inputs, masker_model, conditioning=None):
        embeds = self.forward(inputs)
        seq_length = inputs["input_values"].shape[1]

        weights = masker_model(embeds, seq_length, conditioning)

        inputs_to_use = inputs.copy()
        inputs_to_use["input_values"] = (inputs_to_use["input_values"] * weights)
        masked_embeds = self.forward(inputs_to_use)

        return masked_embeds, weights