import os
import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
from torch.utils.data import DataLoader
import lightning as L
from tqdm.auto import tqdm

from models import get_sep_token, TRAINING_PHASES, TextClaimVerificationModel
from data import DATASETS, get_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="")
    parser.add_argument("--model-card", type=str, required=True, help="Text encoder backbone")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="Training dataset")
    parser.add_argument("--data-dir", type=str, default="./_datasets", help="Directory where the dataset is located/downloaded")
    parser.add_argument("--prediction-json", type=str, default="./_predictions.json", help="JSON file containing model predictions on test set")
    parser.add_argument("--out-path", type=str, default="./evaluation.json", help="JSON file where the results are saved")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()
    
    L.seed_everything(args.seed)

    sep_tok, sep_tok_id = get_sep_token(args.model_card)
    ds_test, num_classes = get_dataset(args.dataset, args.data_dir, sep_tok, ["test"])

    with open(args.prediction_json, "r") as f:
        all_predictions = json.load(f)

    preds = []
    labels = []
    for i, (pred, (text, label)) in (pbar := tqdm(enumerate(zip(all_predictions, ds_test)))):
        preds.append(np.argmax(pred["logits"]))
        labels.append(label)
        # pbar.set_description(f"Accuracy: {(correct / (i+1))*100:.2f}")

    with open(args.out_path, "w") as f:
        json.dump({
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            # "recall": recall_score(labels, preds),
            # "precision": precision_score(labels, preds),
        }, f)

