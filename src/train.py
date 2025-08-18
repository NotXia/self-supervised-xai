import os
import argparse

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models import get_sep_token, TRAINING_PHASES, TextClaimVerificationModel
from data import DATASETS, get_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--model-card", type=str, required=True, help="Text encoder backbone")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="Training dataset")
    parser.add_argument("--phase", type=str, required=True, choices=TRAINING_PHASES, help="Training phase")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="./_datasets", help="Directory where the dataset is located/downloaded")
    parser.add_argument("--model-dir", type=str, default="./_model", help="Directory where the weights and logs are saved into")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()
    
    L.seed_everything(args.seed)

    sep_tok, sep_tok_id = get_sep_token(args.model_card)
    ds_train, ds_val, num_classes = get_dataset(args.dataset, args.data_dir, sep_tok, ["train", "validation"])

    model = TextClaimVerificationModel(args.model_card, out_classes=num_classes, training_phase=args.phase)

    trainer = L.Trainer(
        max_epochs = args.epochs,
        logger = CSVLogger(os.path.join(args.model_dir, "logs")),
        log_every_n_steps = 1,
        callbacks = [
            ModelCheckpoint(dirpath=os.path.join(args.model_dir, "checkpoints"), save_top_k=1, save_last=True, monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        devices = args.gpus
    )

    trainer.fit(
        model,
        DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2), 
        DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    )