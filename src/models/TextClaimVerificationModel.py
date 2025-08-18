import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional import accuracy

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from .utils import get_sep_token

from typing import Literal, Optional


TRAINING_PHASES = ("classifier", "finetune", "masker")



class _TextEncoderModel(L.LightningModule):
    def __init__(self, model_card, max_seq_length, sep_tok_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_card, 
            max_length = max_seq_length,
            torch_dtype = torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_card, padding_side="left")
        self.max_seq_length = max_seq_length
        self.sep_tok_id = sep_tok_id
        self.hidden_size = self.encoder.config.hidden_size

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding = "max_length",
            max_length = self.max_seq_length,
            truncation = True,
            return_tensors = "pt",
        ).to(self.device)

    def __get_embeddings(
        self,
        input_ids,
        last_hidden_states,
        attention_mask,
        sep_token_id,
    ):
        batch_size, _, hidden_size = last_hidden_states.shape

        sequence_lengths = attention_mask.sum(dim=1) - 1
        pool_selector = torch.logical_and(attention_mask == 1, input_ids == sep_token_id)

        nums_sents = pool_selector.sum(dim=1)

        out_embs = torch.zeros(batch_size, torch.max(nums_sents), hidden_size).to(self.device)
        out_mask = torch.ones(batch_size, 1, torch.max(nums_sents), 1).to(self.device)
        for b in range(batch_size):
            embs = last_hidden_states[b][pool_selector[b]]
            out_embs[b, :len(embs)] = embs
            out_mask[b, :, :len(embs)] = 0.0

        return ( out_embs, out_mask, nums_sents )

    def forward(self, inputs):
        encoder_embeds = self.encoder(**inputs).last_hidden_state
        return self.__get_embeddings(
            inputs.input_ids,
            encoder_embeds, 
            inputs.attention_mask,
            self.sep_tok_id,
        )


class _MaskerModel(L.LightningModule):
    def __init__(
        self, 
        hidden_size, 
        num_heads = 4, 
        num_layers = 4, 
        ff_size = 2048, 
        activation = "gelu", 
        dropout = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.activation = activation
        self.dropout = dropout

        self.transformers = BertEncoder(BertConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_layers,
            num_attention_heads = num_heads,
            intermediate_size = ff_size,
            hidden_act = activation,
            hidden_dropout_prob = dropout,
            attention_probs_dropout_prob = dropout,
        ))
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, embeds, attn_masks, nums_sents):
        embeds_masker = self.transformers(
            embeds, 
            torch.repeat_interleave(attn_masks, repeats=self.num_heads, dim=1)
        ).last_hidden_state
        batch_size, seq_length, _ = embeds_masker.shape
        
        # sents_mask = torch.ones(batch_size, seq_length).to(self.device)
        new_masks = torch.ones_like(attn_masks)
        
        for b in range(batch_size):
            # sents_mask[b, :nums_sents[b]] = 1 - F.sigmoid(self.masker_linear(embeds_masker[b, :nums_sents[b]])[:, 0])
            # new_masks[b, :, :nums_sents[b], 0] = sents_mask[b, :nums_sents[b]]
            new_masks[b, 0, :nums_sents[b], 0] = 1 - F.sigmoid(self.linear(embeds_masker[b, :nums_sents[b]])[:, 0])

        return new_masks


class _ClassifierModel(L.LightningModule):
    def __init__(
        self, 
        out_classes,
        hidden_size, 
        num_heads = 4, 
        num_layers = 4, 
        ff_size = 2048, 
        activation = "gelu", 
        dropout = 0.1
    ):
        super().__init__()

        self.out_classes = out_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.activation = activation
        self.dropout = dropout

        self.transformers = BertEncoder(BertConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_layers,
            num_attention_heads = num_heads,
            intermediate_size = ff_size,
            hidden_act = activation,
            hidden_dropout_prob = dropout,
            attention_probs_dropout_prob = dropout,
        ))
        self.linear = torch.nn.Linear(hidden_size, out_classes)

    def forward(self, embeds, attn_masks):
        embeds_cls = self.transformers(
            embeds, 
            torch.repeat_interleave(attn_masks, repeats=self.num_heads, dim=1)
        ).last_hidden_state
        logits = self.linear(embeds_cls[:, 0])
        return logits



class TextClaimVerificationModel(L.LightningModule):
    def __init__(
        self, 
        text_encoder_card, 
        training_phase: Optional[Literal[TRAINING_PHASES]] = None, 
        max_seq_length = 8192, 
        out_classes = 2
    ):
        super().__init__()
        self.__training_phase = training_phase
        self.out_classes = out_classes
        self.max_seq_length = max_seq_length

        self.sep_tok_id = get_sep_token(text_encoder_card)[1]
        self.text_encoder = _TextEncoderModel(text_encoder_card, self.max_seq_length, self.sep_tok_id)
        self.hidden_size = self.text_encoder.hidden_size

        self.masker = _MaskerModel(self.hidden_size)
        self.classifier = _ClassifierModel(self.out_classes, self.hidden_size)


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
                    lr = 2e-4
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
                    lr = 1e-5
                )
            case "masker":
                self.text_encoder.eval().requires_grad_(False)
                self.masker.train().requires_grad_(True)
                self.classifier.eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [
                        *self.masker.parameters(),
                    ], 
                    lr = 2e-4
                )
            case _:
                optimizer = None
        return optimizer


    def preprocess(self, texts):
        return self.text_encoder.preprocess(texts)


    def forward(self, inputs):
        embeds_enc, masks, nums_sents = self.text_encoder(inputs)
        if (self.__training_phase is None) or (self.__training_phase == "masker"):
            new_masks = self.masker(embeds_enc, masks, nums_sents)
        else:
            new_masks = masks
        logits = self.classifier(embeds_enc, new_masks)

        return logits, (1-new_masks).squeeze(1, 3)


    def __loss(self, logits, labels, masks, log_prefix):
        loss = 0

        loss_cls = F.cross_entropy(logits, labels)
        self.log(f"{log_prefix}_loss_cls", loss_cls, sync_dist=True)
        loss += loss_cls

        loss_mask = torch.sum(torch.linalg.norm(masks, ord=2, dim=1)) if self.__training_phase == "masker" else 0
        self.log(f"{log_prefix}_loss_mask", loss_mask, sync_dist=True)
        loss += loss_mask

        self.log(f"{log_prefix}_loss", loss, sync_dist=True)
        return loss


    def training_step(self, batch, batch_idx):
        texts, labels = batch

        inputs = self.preprocess(texts)
        logits, sents_masks = self.forward(inputs)

        loss = self.__loss(logits, labels, sents_masks, "train")

        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        self.log("train_acc", acc, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        texts, labels = batch

        inputs = self.preprocess(texts)
        logits, sents_masks = self.forward(inputs)

        loss = self.__loss(logits, labels, sents_masks, "val")

        acc = accuracy(logits, labels, task="multiclass", num_classes=self.out_classes)
        self.log("val_acc", acc, sync_dist=True)

        return loss
