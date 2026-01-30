import math

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional import accuracy, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from .utils.masker import TextMaskerModel
from .utils.text_encoder import TextEncoderModel

from typing import Literal, Optional


TRAINING_PHASES = ("classifier", "finetune", "masker")


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



class MultipleChoiceQAModel(L.LightningModule):
    def __init__(
        self, 
        text_encoder_card, 
        training_phase: Optional[Literal[TRAINING_PHASES]] = None, 
        require_mask = True,
        max_seq_length = 4096,
        out_classes = 3,
        classifier_lr = 1e-4,
        finetune_lr = 1e-5,
        masker_lr = 1e-4,
        loss_binary_weight = 0.01,
        loss_norm_weight = 0.05,
    ):
        super().__init__()
        self.__training_phase = training_phase
        self.out_classes = out_classes
        self.max_seq_length = max_seq_length

        self.require_mask = require_mask

        self.classifier_lr = classifier_lr
        self.finetune_lr = finetune_lr
        self.masker_lr = masker_lr

        self.loss_binary_weight = loss_binary_weight
        self.loss_norm_weight = loss_norm_weight

        self.text_encoder = TextEncoderModel(
            text_encoder_card, 
            self.max_seq_length, 
        )
        self.masker = TextMaskerModel(
            self.text_encoder.hidden_size,
        )
        self.classifier = _ClassifierModel(
            self.out_classes, 
            self.text_encoder.hidden_size,
        )


    def configure_optimizers(self):
        match self.__training_phase:
            case "classifier":
                self.text_encoder.eval().requires_grad_(False)
                self.masker.eval().requires_grad_(False)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [
                        *self.classifier.parameters(),
                    ], 
                    lr = self.classifier_lr
                )
            case "finetune":
                self.text_encoder.train().requires_grad_(True)
                self.masker.eval().requires_grad_(False)
                self.classifier.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [ 
                        *self.classifier.parameters(),
                        *self.text_encoder.parameters() 
                    ], 
                    lr = self.finetune_lr
                )
            case "masker":
                self.text_encoder.eval().requires_grad_(False)
                self.masker.train().requires_grad_(True)
                self.classifier.eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [
                        *self.masker.parameters(),
                    ], 
                    lr = self.masker_lr
                )
            case _:
                optimizer = None
        return optimizer


    def preprocess(self, context, question, options):
        texts = [
            f"[CONTEXT] {context[i]} [QUESTION] {question[i]} {' '.join([f'[OPTION{j}] {o[i]}' for j, o in enumerate(options)])}"
            for i in range(len(context))
        ]
        return self.text_encoder.preprocess(texts)


    def __forward(self, inputs):
        sequence_lengths = inputs["attention_mask"].sum(dim=1)
        
        if (self.__training_phase is None and self.require_mask) or (self.__training_phase == "masker"):
            embeds_enc, weights = self.text_encoder.embed_with_masker(inputs, self.masker)
        else:
            embeds_enc = self.text_encoder(inputs)
            weights = None
        logits = self.classifier(embeds_enc.mean(dim=1))

        return logits, weights, sequence_lengths

    def forward(self, inputs):
        logits, weights, _ = self.__forward(inputs)
        return logits, weights


    def __loss(self, logits, labels, weights, sequence_lengths, log_prefix):
        loss = 0.0

        loss_cls = F.cross_entropy(logits, labels)
        self.log(f"{log_prefix}_loss_cls", loss_cls, sync_dist=True)
        loss += loss_cls

        loss_mask = 0.0
        if self.__training_phase == "masker":
            batch_size, seq_len, _ = weights.shape
            loss_mask += self.loss_binary_weight * (1/batch_size) * sum( torch.mean((weights[b, :] * (1-weights[b, :]))**2) for b in range(batch_size) )
            loss_mask += self.loss_norm_weight * (1/batch_size) * sum( torch.linalg.norm(weights[b, :sequence_lengths[b]]) for b in range(batch_size) )
            
        self.log(f"{log_prefix}_loss_mask", loss_mask, sync_dist=True)
        loss += loss_mask

        self.log(f"{log_prefix}_loss", loss, sync_dist=True)
        return loss


    def training_step(self, batch, batch_idx):
        context, question, options, labels = batch

        inputs = self.preprocess(context, question, options)
        logits, sents_weights, sequence_lengths = self.__forward(inputs)

        loss = self.__loss(logits, labels, sents_weights, sequence_lengths, "train")

        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("train_acc", acc, sync_dist=True)
        self.log("train_f1macro", f1, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        context, question, options, labels = batch

        inputs = self.preprocess(context, question, options)
        logits, sents_weights, sequence_lengths = self.__forward(inputs)

        loss = self.__loss(logits, labels, sents_weights, sequence_lengths, "val")

        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        f1 = f1_score(logits, labels, task="multiclass", average="macro", num_classes=self.out_classes)
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_f1macro", f1, sync_dist=True)

        return loss
