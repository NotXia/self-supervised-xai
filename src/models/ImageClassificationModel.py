import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional import accuracy, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from transformers import ViTImageProcessor, ViTModel, CLIPImageProcessor, AutoImageProcessor, AutoModel
import torchvision

from .utils.utils import get_sep_token, get_vit_config
from .utils.masker import ImageMaskerModel
from .utils.image_encoder import ImageEncoderModel

from typing import Literal, Optional


TRAINING_PHASES = ("classifier", "finetune", "masker")


class _ClassifierModel(L.LightningModule):
    def __init__(
        self, 
        out_classes,
        hidden_size, 
        dropout = 0.5,
    ):
        super().__init__()

        self.out_classes = out_classes
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_size, out_classes),
        )

    def forward(self, embeds):
        logits = self.head(embeds.mean(1))
        return logits




class ImageClassificationModel(L.LightningModule):
    def __init__(
        self, 
        image_encoder_card,
        training_phase: Optional[Literal[TRAINING_PHASES]] = None, 
        require_mask = True,
        out_classes = 2,
        use_contrastive_loss = False,
        classifier_lr = 1e-4,
        finetune_lr = 1e-5,
        masker_lr = 1e-4,
        loss_binary_weight = 0.05,
        loss_norm_weight = 0.05,
        skip_masker_relu = False,
        skip_masker_sigmoid = False,
        skip_masker_rescale = False,
        skip_loss_penalty_binary = False,
        skip_loss_penalty_magnitude = False
    ):
        super().__init__()
        self.__training_phase = training_phase
        self.out_classes = out_classes
        self.classifier_lr = classifier_lr
        self.finetune_lr = finetune_lr
        self.masker_lr = masker_lr
        self.loss_binary_weight = loss_binary_weight
        self.loss_norm_weight = loss_norm_weight

        self.image_encoder = ImageEncoderModel(
            image_encoder_card
        )
        self.image_hidden_size = self.image_encoder.hidden_size
        self.image_masker = ImageMaskerModel(
            self.image_hidden_size,
            in_resolution = self.image_encoder.out_resolution,
            out_resolution = self.image_encoder.in_resolution,
            skip_relu = skip_masker_relu,
            skip_sigmoid = skip_masker_sigmoid,
            skip_rescale = skip_masker_rescale,
        )

        self.classifier = _ClassifierModel(
            self.out_classes, 
            self.image_hidden_size,
        )
        self.require_mask = require_mask
        self.use_contrastive_loss = use_contrastive_loss

        # For ablation
        self.skip_loss_penalty_binary = skip_loss_penalty_binary
        self.skip_loss_penalty_magnitude = skip_loss_penalty_magnitude


    def configure_optimizers(self):
        match self.__training_phase:
            case "classifier":
                self.image_encoder.eval().requires_grad_(False)
                self.image_masker.eval().requires_grad_(False)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [
                        *self.classifier.parameters(),
                    ], 
                    lr = self.classifier_lr
                )
            case "finetune":
                self.image_encoder.train().requires_grad_(True)
                self.image_masker.eval().requires_grad_(False)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [ 
                        *self.image_encoder.parameters(),
                        *self.classifier.parameters(),
                    ], 
                    lr = self.finetune_lr
                )
            case "masker":
                self.image_encoder.eval().requires_grad_(False)
                self.image_masker.train().requires_grad_(True)
                self.classifier.eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [
                        *self.image_masker.parameters()
                    ], 
                    lr = self.masker_lr,
                )
            case _:
                optimizer = None
        return optimizer


    def preprocess(self, images):
        return self.image_encoder.preprocess(images)


    def forward(self, image_inputs, training=False):
        image_embeds, image_hidden_states = self.image_encoder(image_inputs)

        image_weights = None
        if (self.__training_phase is None and self.require_mask) or (self.__training_phase == "masker"):
            image_embeds_masked, image_weights = self.image_encoder.embed_with_masker(image_inputs, self.image_masker)
            logits_masked = self.classifier(image_embeds_masked)

            if self.__training_phase == "masker":
                logits_orig = self.classifier(image_embeds)
                logits = (logits_orig, logits_masked)
            else:
                logits = logits_masked
        else:
            logits = self.classifier(image_embeds)

        if training:
            return logits, None, image_weights
        else:
            return logits, image_weights


    def __loss(self, logits, logits_contrastive, labels, image_weights, log_prefix):
        loss = 0.0

        if self.__training_phase == "masker":
            loss_cls = F.cross_entropy(logits[1], labels)
        else:
            loss_cls = F.cross_entropy(logits, labels)
        self.log(f"{log_prefix}_loss_cls", loss_cls, sync_dist=True)
        loss += loss_cls

        loss_mask = 0.0
        loss_contrastive = 0.0
        if self.__training_phase == "masker":
            ce_orig = F.cross_entropy(logits[0], labels, reduction="none")
            ce_masked = F.cross_entropy(logits[1], labels, reduction="none")
            bs_size, _, weights_h, weights_w = image_weights.shape

            if not self.skip_loss_penalty_binary:
                loss_mask += (1/bs_size) * sum( max(0, min((1/(ce_masked[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean((image_weights[b] * (1-image_weights[b]))**2) for b in range(bs_size) )
            if not self.skip_loss_penalty_magnitude:
                loss_mask += (1/bs_size) * sum( max(0, min((1/(ce_masked[b]-ce_orig[b]+1e-16)), 1.0)) * torch.mean(image_weights[b])**2 for b in range(bs_size))

        self.log(f"{log_prefix}_loss_mask", loss_mask, sync_dist=True, prog_bar=True)
        loss += loss_mask

        self.log(f"{log_prefix}_loss", loss, sync_dist=True, prog_bar=True)
        return loss


    def training_step(self, batch, batch_idx):
        images, labels = batch

        image_inputs = self.preprocess(images)
        logits, logits_contrastive, image_weights = self.forward(image_inputs, training=True)

        loss = self.__loss(logits, logits_contrastive, labels, image_weights, "train")

        logits = logits[1] if self.__training_phase == "masker" else logits
        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("train_acc", acc, sync_dist=True, prog_bar=True)
        self.log("train_f1macro", f1, sync_dist=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch

        image_inputs = self.preprocess(images)
        logits, logits_contrastive, image_weights = self.forward(image_inputs, training=True)

        loss = self.__loss(logits, logits_contrastive, labels, image_weights, "val")

        logits = logits[1] if self.__training_phase == "masker" else logits
        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("val_acc", acc, sync_dist=True, prog_bar=True)
        self.log("val_f1macro", f1, sync_dist=True, prog_bar=True)

        return loss
