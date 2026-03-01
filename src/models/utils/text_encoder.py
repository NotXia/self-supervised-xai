import math

import torch
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel, AutoConfig

from typing import Literal, Optional



class _Qwen3Encoder(L.LightningModule):
    def __init__(self, model_card, max_seq_length):
        super().__init__()
        self.model_card = model_card
        self.max_seq_length = max_seq_length

        self.encoder = AutoModel.from_pretrained(
            self.model_card, 
            max_length = self.max_seq_length,
            dtype = torch.float32
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_card, padding_side="left")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding = "max_length",
            max_length = self.max_seq_length,
            truncation = True,
            # add_special_tokens = False,
            return_tensors = "pt",
        ).to(self.device)

    def embed(self, input_ids):
        return self.encoder.embed_tokens(input_ids)

    def forward(self, inputs_embeds, attn_masks):
        encoder_embeds = self.encoder(
            inputs_embeds = inputs_embeds,
            attn_masks = attn_masks
        ).last_hidden_state
        return encoder_embeds


class _RoBERTaEncoder(L.LightningModule):
    def __init__(self, model_card, max_seq_length):
        super().__init__()
        self.model_card = model_card
        self.max_seq_length = max_seq_length

        self.encoder = AutoModel.from_pretrained(
            self.model_card, 
            max_length = self.max_seq_length,
            dtype = torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding = "max_length",
            max_length = self.max_seq_length,
            truncation = True,
            add_special_tokens = True,
            return_tensors = "pt",
        ).to(self.device)

    def embed(self, input_ids):
        return self.encoder.embeddings(input_ids)

    def forward(self, inputs_embeds, attn_masks):
        encoder_embeds = self.encoder(
            inputs_embeds = inputs_embeds,
            attention_mask = attn_masks
        ).last_hidden_state

        return encoder_embeds


class _LongformerEncoder(L.LightningModule):
    def __init__(self, model_card, max_seq_length):
        super().__init__()
        self.model_card = model_card
        self.max_seq_length = max_seq_length

        self.encoder = AutoModel.from_pretrained(
            self.model_card, 
            max_length = self.max_seq_length,
            dtype = torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding = "max_length",
            max_length = self.max_seq_length,
            truncation = True,
            add_special_tokens = True,
            return_tensors = "pt",
        ).to(self.device)

    def embed(self, input_ids):
        return self.encoder.embeddings(input_ids)

    def forward(self, inputs_embeds, attn_masks):
        encoder_embeds = self.encoder(
            inputs_embeds = inputs_embeds,
            attention_mask = attn_masks
        ).last_hidden_state

        return encoder_embeds



class TextEncoderModel(L.LightningModule):
    def __init__(self, model_card, max_seq_length):
        super().__init__()
        match model_card:
            case "Qwen/Qwen3-Embedding-0.6B":
                self._model = _Qwen3Encoder(model_card, max_seq_length)
                self.hidden_size = 1024
            case "FacebookAI/roberta-base":
                self._model = _RoBERTaEncoder(model_card, max_seq_length)
                self.hidden_size = 768
            case "allenai/longformer-base-4096":
                self._model = _LongformerEncoder(model_card, max_seq_length)
                self.hidden_size = 768

    def preprocess(self, texts):
        return self._model.preprocess(texts)

    def embed(self, input_ids):
        return self._model.embed(input_ids)

    def forward(self, inputs):
        inputs_embedded = self.embed(inputs["input_ids"]) # Embed input token indices (static embedding only)
        attn_masks = inputs["attention_mask"]
        return self._model(inputs_embedded, attn_masks)

    def embed_with_masker(self, inputs, masker_model, conditioning=None):
        inputs_embedded = self.embed(inputs["input_ids"]) # Embed input token indices (static embedding only)
        attn_masks = inputs["attention_mask"]

        embeds_enc = self._model(inputs_embedded, attn_masks)

        # Compute attribution and recompute embeddings accordingly
        weights = masker_model(embeds_enc, attn_masks, conditioning)
        embeds_enc = self._model((inputs_embedded * weights), attn_masks)

        return embeds_enc, weights