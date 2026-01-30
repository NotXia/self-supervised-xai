import os
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models import TRAINING_PHASES, get_model
from data import get_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--checkpoint", type=str, required=False, help="Checkpoint to load")
    parser.add_argument("--gpus", type=str, default="0", help="GPUs indices")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed-precision", action="store_true", default=False, help="Use b16 mixed precision")
    parser.add_argument("--phase", type=str, required=True, choices=TRAINING_PHASES, help="Training phase")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="Directory where the dataset is located/downloaded")
    parser.add_argument("--model-dir", type=str, required=False, 
        help="Directory where the weights and logs are saved into. If not set, the same directory of the model config will be used")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()
    
    L.seed_everything(args.seed)

    # Get model configuration
    model_config = json.load( open(args.model_config, "r") )
    config_dir = Path(args.model_config).parent

    # Prepare output directory
    model_dir = os.path.join(config_dir, args.phase) if args.model_dir is None else args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Prepare trainer
    trainer = L.Trainer(
        max_epochs = args.epochs,
        logger = CSVLogger(os.path.join(model_dir, "logs")),
        log_every_n_steps = 1,
        callbacks = [
            # ModelCheckpoint(dirpath=os.path.join(model_dir, "checkpoints"), save_top_k=1, save_last=True, monitor="val_loss", mode="min"),
            # EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(dirpath=os.path.join(model_dir, "checkpoints"), save_top_k=1, save_last=True, monitor="val_acc", mode="max"),
            EarlyStopping(monitor="val_acc", patience=10, mode="max"),

        ],
        strategy = "ddp_find_unused_parameters_true",
        devices = args.gpus,
        accumulate_grad_batches = args.grad_accum,
        precision = "bf16-mixed" if args.mixed_precision else None
    )


    # Load datasets
    ds_train, ds_val, num_classes = get_dataset(
        dataset_name = model_config["dataset_name"],
        data_dir = args.data_dir, 
        splits = ["train", "validation"],
        seed = args.seed,
        **model_config["dataset_args"]
    )


    # Load model
    model = get_model(
        model_name = model_config["model_name"], 
        checkpoint_path = args.checkpoint, 
        training_phase = args.phase,
        **model_config["model_args"]
    )


    # Dump arguments
    with open(os.path.join(model_dir, "_metadata.json"), "w") as f:
        json.dump({
            "arguments": args.__dict__,
            "config": model_config
        }, f, indent=4)


    # Training
    trainer.fit(
        model,
        DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2), 
        DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    )


    # Dump updated metadata
    with open(os.path.join(model_dir, "_metadata.json"), "w") as f:
        json.dump({
            "arguments": args.__dict__,
            "best_checkpoint": os.path.relpath(trainer.checkpoint_callback.best_model_path, start=os.path.abspath(config_dir)),
            "best_checkpoint_abs": trainer.checkpoint_callback.best_model_path,
            "config": model_config
        }, f, indent=4)
