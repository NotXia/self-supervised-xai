import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
import lightning as L
from tqdm.auto import tqdm

from models import get_sep_token, TRAINING_PHASES, TextClaimVerificationModel
from data import DATASETS, get_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="")
    parser.add_argument("--model-card", type=str, required=True, help="Text encoder backbone")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="Evaluation dataset")
    parser.add_argument("--checkpoint", type=str, required=False, help="Checkpoint to load")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--no-masker", action="store_true", default=False)
    parser.add_argument("--data-dir", type=str, default="./_datasets", help="Directory where the dataset is located/downloaded")
    parser.add_argument("--out-path", type=str, default="./_predictions.json", help="JSON file where the results are saved")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()
    
    L.seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    sep_tok, sep_tok_id = get_sep_token(args.model_card)
    ds_test, num_classes = get_dataset(args.dataset, args.data_dir, sep_tok, ["test"])
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = TextClaimVerificationModel.load_from_checkpoint(
        args.checkpoint,
        map_location = "cpu",
        text_encoder_card = args.model_card, 
        require_mask = not args.no_masker,
        out_classes = num_classes, 
    ).eval().cuda()

    predictions = []

    with torch.no_grad():
        for inputs, labels in (pbar := tqdm(dl_test)):
            inputs = model.preprocess(inputs)
            logits, sents_weights = model(inputs)
            for b in range(len(logits)):
                predictions.append({
                    "logits": logits[b].tolist(),
                    "sents_weights": sents_weights[b].squeeze(1).tolist() if sents_weights is not None else None
                })

    with open(args.out_path, "w") as f:
        json.dump(predictions, f, indent=4)

