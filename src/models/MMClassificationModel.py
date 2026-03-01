import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional import accuracy, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from .utils.text_encoder import TextEncoderModel
from .utils.image_encoder import ImageEncoderModel
from .utils.masker import ImageMaskerModel, TextMaskerModel

from typing import Literal, Optional


TRAINING_PHASES = ("classifier", "finetune", "masker")



class _FusionModel(L.LightningModule):
    def __init__(
        self, 
        hidden_size_text, 
        hidden_size_image, 
    ):
        super().__init__()
        self.hidden_size = hidden_size_text
        self.num_heads = 12

        self.linear = torch.nn.Linear(hidden_size_image, hidden_size_text)
        self.transformers = BertEncoder(BertConfig(
            hidden_size = self.hidden_size,
            num_hidden_layers = 6,
            num_attention_heads = self.num_heads,
            intermediate_size = 2048,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.5,
            attention_probs_dropout_prob = 0.5,
            _attn_implementation = "eager"
        ))

    def forward(self, embeds_text, embeds_image, text_attn_masks):
        # Prepare mask for transformer
        image_attn_masks = torch.ones(embeds_image.shape[0], embeds_image.shape[1]).to(self.device)
        attn_masks = torch.cat([image_attn_masks, text_attn_masks], dim=1)
        attn_masks = attn_masks.unsqueeze(1).unsqueeze(3)
        attn_masks = torch.repeat_interleave(attn_masks, repeats=self.num_heads, dim=1)

        # Compute token-wise embeddings
        embeds = self.transformers(
            torch.cat([embeds_image, embeds_text], dim=1), 
            attn_masks
        ).last_hidden_state

        return embeds


class _ClassifierModel(L.LightningModule):
    def __init__(
        self, 
        out_classes,
        hidden_size, 
    ):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_classes),
        )

    def forward(self, embeds):
        logits = self.head(embeds)
        return logits



class MMClassificationModel(L.LightningModule):
    def __init__(
        self, 
        text_encoder_card, 
        image_encoder_card, 
        training_phase: Optional[Literal[TRAINING_PHASES]] = None, 
        require_mask = True,
        text_max_seq_length = 4096,
        out_classes = 3,
        classifier_lr = 1e-4,
        finetune_lr = 1e-5,
        masker_lr = 1e-4,
        loss_binary_text_weight = 0.01,
        loss_norm_text_weight = 0.05,
        loss_binary_image_weight = 0.01,
        loss_norm_image_weight = 0.05,
    ):
        super().__init__()
        self.__training_phase = training_phase
        self.out_classes = out_classes
        self.text_max_seq_length = text_max_seq_length

        self.require_mask = require_mask

        self.classifier_lr = classifier_lr
        self.finetune_lr = finetune_lr
        self.masker_lr = masker_lr

        self.loss_binary_text_weight = loss_binary_text_weight
        self.loss_norm_text_weight = loss_norm_text_weight
        self.loss_binary_image_weight = loss_binary_image_weight
        self.loss_norm_image_weight = loss_norm_image_weight

        # Text branch
        self.text_encoder = TextEncoderModel(
            text_encoder_card, 
            self.text_max_seq_length, 
        )
        self.text_masker = TextMaskerModel(
            self.text_encoder.hidden_size,
            max_seq_length = self.text_max_seq_length
        )

        # Image branch
        self.image_encoder = ImageEncoderModel(
            image_encoder_card
        )
        self.image_masker_conditioning = nn.Linear(self.text_encoder.hidden_size, self.image_encoder.hidden_size)
        self.image_masker = ImageMaskerModel(
            self.image_encoder.hidden_size,
            in_resolution = self.image_encoder.out_resolution,
            out_resolution = self.image_encoder.in_resolution
        )

        self.fusion = _FusionModel(
            self.text_encoder.hidden_size,
            self.image_encoder.hidden_size, 
        )
        
        self.classifier = _ClassifierModel(
            self.out_classes, 
            self.fusion.hidden_size,
        )


    def configure_optimizers(self):
        match self.__training_phase:
            case "classifier":
                self.text_encoder.eval().requires_grad_(False)
                self.text_masker.eval().requires_grad_(False)
                self.image_encoder.eval().requires_grad_(False)
                self.image_masker.eval().requires_grad_(False)
                self.fusion.train().requires_grad_(True)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [
                        *self.fusion.parameters(),
                        *self.classifier.parameters(),
                    ], 
                    lr = self.classifier_lr
                )
            case "finetune":
                self.text_encoder.train().requires_grad_(True)
                self.text_masker.eval().requires_grad_(False)
                self.image_encoder.train().requires_grad_(True)
                self.image_masker.eval().requires_grad_(False)
                self.fusion.train().requires_grad_(True)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [ 
                        *self.text_encoder.parameters(),
                        *self.image_encoder.parameters(),
                        *self.fusion.parameters(),
                        *self.classifier.parameters(),
                    ], 
                    lr = self.finetune_lr,
                )
            case "masker":
                self.text_encoder.eval().requires_grad_(False)
                self.text_masker.train().requires_grad_(True)
                self.image_encoder.eval().requires_grad_(False)
                self.image_masker.train().requires_grad_(True)
                self.fusion.eval().requires_grad_(False)
                self.classifier.eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [
                        *self.text_masker.parameters(),
                        *self.image_masker.parameters(),
                    ], 
                    lr = self.masker_lr
                )
            case _:
                optimizer = None
        return optimizer


    def preprocess(self, image, text):
        return self.image_encoder.preprocess(image), self.text_encoder.preprocess(text)


    def forward(self, image_inputs, text_inputs, training=False):
        text_sequence_lengths = text_inputs["attention_mask"].sum(dim=1)

        if (self.__training_phase is None and self.require_mask) or (self.__training_phase == "masker"):
            # Embed with attributions
            text_embeds = self.text_encoder(text_inputs)
            image_embeds, _ = self.image_encoder(image_inputs)
            fusion_embeds = self.fusion(text_embeds, image_embeds, text_inputs["attention_mask"]).mean(dim=1)

            image_conditioning = self.image_masker_conditioning(fusion_embeds)
            text_embeds_masked,  text_weights  =  self.text_encoder.embed_with_masker(text_inputs,  self.text_masker, conditioning=None)
            image_embeds_masked, image_weights = self.image_encoder.embed_with_masker(image_inputs, self.image_masker, conditioning=None)

            if self.__training_phase == "masker":
                fusion_embeds_text = self.fusion(text_embeds_masked, image_embeds, text_inputs["attention_mask"]).mean(dim=1)
                fusion_embeds_image = self.fusion(text_embeds, image_embeds_masked, text_inputs["attention_mask"]).mean(dim=1)
                fusion_embeds_both = self.fusion(text_embeds_masked, image_embeds_masked, text_inputs["attention_mask"]).mean(dim=1)
                logits_orig = self.classifier(fusion_embeds)
                logits_text = self.classifier(fusion_embeds_text)
                logits_image = self.classifier(fusion_embeds_image)
                logits_both = self.classifier(fusion_embeds_both)
                logits = (logits_text, logits_image, logits_both, logits_orig)
            else:
                fusion_embeds = self.fusion(text_embeds_masked, image_embeds_masked, text_inputs["attention_mask"]).mean(dim=1)
                logits = self.classifier(fusion_embeds)
        else:
            # Normal embedding
            text_embeds = self.text_encoder(text_inputs)
            image_embeds, _ = self.image_encoder(image_inputs)
            text_weights, image_weights = None, None

            fusion_embeds = self.fusion(text_embeds, image_embeds, text_inputs["attention_mask"]).mean(dim=1)
            logits = self.classifier(fusion_embeds)

        if training:
            return logits, text_sequence_lengths, text_weights, image_weights
        else:
            return logits, text_weights, image_weights


    def __loss(self, logits, labels, text_sequence_lengths, text_weights, image_weights, log_prefix):
        loss = 0.0
        if self.__training_phase == "masker":
            batch_size = len(logits[0])
        else:
            batch_size = len(logits)

        if self.__training_phase == "masker":
            loss_cls = (3*F.cross_entropy(logits[0], labels) + F.cross_entropy(logits[1], labels) + F.cross_entropy(logits[2], labels)) / 3
        else:
            loss_cls = F.cross_entropy(logits, labels)
        self.log(f"{log_prefix}_loss_cls", loss_cls, sync_dist=True)
        loss += loss_cls

        loss_mask = 0.0
        loss_mask_text = 0.0
        loss_mask_image = 0.0
        loss_contrastive = 0.0
        if self.__training_phase == "masker":
            _, _, weights_h, weights_w = image_weights.shape

            ce_orig = F.cross_entropy(logits[3], labels, reduction="none")

            ce_masked_image = F.cross_entropy(logits[1], labels, reduction="none")
            loss_mask_image += (1/batch_size) * sum( max(0, min((1/(ce_masked_image[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean((image_weights[b] * (1-image_weights[b]))**2) for b in range(batch_size) )
            loss_mask_image += (1/batch_size) * sum( max(0, min((1/(ce_masked_image[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean(image_weights[b])**2 for b in range(batch_size))

            ce_orig_text = F.cross_entropy(logits[3], labels, reduction="none")
            ce_masked_text = F.cross_entropy(logits[0], labels, reduction="none")
            loss_mask_text += (1/batch_size) * sum( max(0, min((1/(ce_masked_text[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean((text_weights[b, :] * (1-text_weights[b, :]))**2) for b in range(batch_size) )
            loss_mask_text += (1/batch_size) * sum( max(0, min((1/(ce_masked_text[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean(text_weights[b, :text_sequence_lengths[b]])**2 for b in range(batch_size) )

            loss_mask += (loss_mask_text + loss_mask_image) / 2
            ce_mask = F.cross_entropy(logits[2], labels)
            loss_mask += max(0, min((1/(ce_mask-torch.mean(ce_orig)+1e-16)), 1.0)) * ((loss_mask_text + loss_mask_image) / 2)

        self.log(f"{log_prefix}_loss_mask_text", loss_mask_text, sync_dist=True, prog_bar=True)
        self.log(f"{log_prefix}_loss_mask_image", loss_mask_image, sync_dist=True, prog_bar=True)
        self.log(f"{log_prefix}_loss_mask", loss_mask, sync_dist=True, prog_bar=True)
        loss += loss_mask

        self.log(f"{log_prefix}_loss", loss, sync_dist=True, prog_bar=True)
        return loss


    def training_step(self, batch, batch_idx):
        image, text, labels = batch

        image_inputs, text_inputs = self.preprocess(image, text)
        logits, text_sequence_lengths, text_weights, image_weights = self.forward(image_inputs, text_inputs, training=True)

        loss = self.__loss(logits, labels, text_sequence_lengths, text_weights, image_weights, "train")

        if self.__training_phase == "masker":
            # acc = (
            #     accuracy(logits[0], labels, task="multiclass", num_classes=self.out_classes)
            #     + accuracy(logits[1], labels, task="multiclass", num_classes=self.out_classes)
            # ) / 2
            # f1 = (
            #     f1_score(logits[0], labels, task="multiclass", average="macro", num_classes=self.out_classes)
            #     + f1_score(logits[1], labels, task="multiclass", average="macro", num_classes=self.out_classes)
            # ) / 2
            acc = accuracy(logits[2], labels, task="multiclass", num_classes=self.out_classes)
            f1 = f1_score(logits[2], labels, task="multiclass", average="macro", num_classes=self.out_classes)
        else:
            acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
            f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("train_acc", acc, sync_dist=True, prog_bar=True)
        self.log("train_f1macro", f1, sync_dist=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        image, text, labels = batch

        image_inputs, text_inputs = self.preprocess(image, text)
        logits, text_sequence_lengths, text_weights, image_weights = self.forward(image_inputs, text_inputs, training=True)

        loss = self.__loss(logits, labels, text_sequence_lengths, text_weights, image_weights, "val")

        if self.__training_phase == "masker":
            # acc = (
            #     accuracy(logits[0], labels, task="multiclass", num_classes=self.out_classes)
            #     + accuracy(logits[1], labels, task="multiclass", num_classes=self.out_classes)
            # ) / 2
            # f1 = (
            #     f1_score(logits[0], labels, task="multiclass", average="macro", num_classes=self.out_classes)
            #     + f1_score(logits[1], labels, task="multiclass", average="macro", num_classes=self.out_classes)
            # ) / 2
            acc = accuracy(logits[2], labels, task="multiclass", num_classes=self.out_classes)
            f1 = f1_score(logits[2], labels, task="multiclass", average="macro", num_classes=self.out_classes)
        else:
            acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
            f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("val_acc", acc, sync_dist=True, prog_bar=True)
        self.log("val_f1macro", f1, sync_dist=True, prog_bar=True)

        return loss
