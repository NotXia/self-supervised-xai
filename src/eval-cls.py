import argparse
import os
import json
import random
import itertools
import math
import warnings
from tqdm.auto import tqdm
from tabulate import tabulate

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import lightning as L

from sklearn.metrics import f1_score, accuracy_score
from torchmetrics.classification import MulticlassJaccardIndex

from models import get_model
from data import get_dataset
from utils.plot import *
from xai import *
from metrics import *




def _tokens_to_scores(model, tokens, attr):
    curr_word, curr_scores = "", []
    all_words, all_scores = [], []

    for t, s in zip(model.text_encoder._model.tokenizer.batch_decode(tokens), attr[0, :, 0]):
        if t == "<s>": continue

        if t[0] != " " and t != "</s>":
            curr_word += t
            curr_scores.append(s)
        else:
            all_words.append( curr_word )
            all_scores.append( float(np.mean(curr_scores)) )
            curr_word, curr_scores = t, [s]

        if t == "</s>": break

    return torch.as_tensor(all_scores)

def eval_text(model, ds_train, ds_test, threshold=0.5):
    methods = [ "ours", "layer-ig", "saliency", "deeplift", "deeplift-shap", "gradient-shap", "guided-backprop" ]
    metrics_accum = {
        method: {
            "iou": MulticlassJaccardIndex(2, average="macro"),
        } for method in methods
    }
    all_preds = []
    all_labels = []

    base_model = get_model_wrapper(model) # Model without explainer

    for i in tqdm(range(len(ds_test))):
        text, label, xai_label = ds_test[i]

        with torch.no_grad():
            inputs = model.preprocess(text)
            logits, _ = model(inputs)
            logits_base = base_model(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.argmax(logits).item()
        pred_base = torch.argmax(logits_base).item()

        tokens = inputs["input_ids"].cpu().squeeze(0)
        tokens = tokens.tolist()

        # Prepare explainers inputs
        inputs = (inputs["input_ids"], inputs["attention_mask"])
        input_embeds = model.text_encoder.embed(inputs[0])
        seq_length = inputs[1].sum().item()

        # Compute attribution maps with our method
        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0]

        # Compute attribution maps with all baselines
        explainer_lig = LayerIntegratedGradientsAttribution(model, baselines=(inputs[0] * 0))
        attr_lig = explainer_lig(inputs[0], target=pred, additional_forward_args=inputs[1])[0]

        explainer_saliency = SaliencyAttribution(model)
        attr_saliency = explainer_saliency(input_embeds, target=pred, additional_forward_args=inputs[1])[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dl = DeepLiftAttribution(model, baselines=(input_embeds * 0.0))
            attr_dl = explainer_dl(input_embeds, target=pred, additional_forward_args=inputs[1])[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baselines_dlshap = torch.cat([ model.text_encoder.embed(model.preprocess(ds_train[i][0])["input_ids"]) for i in range(20) ], dim=0)
            explainer_dlshap = DeepLiftShapAttribution(model, baselines=baselines_dlshap)
            attr_dlshap = explainer_dlshap(input_embeds, target=pred, additional_forward_args=inputs[1].unsqueeze(1))[0]

        baselines_grad_shap = torch.cat([ model.text_encoder.embed(model.preprocess(ds_train[i][0])["input_ids"]) for i in range(20) ], dim=0)
        explainer_grad_shap = GradientShapAttribution(model, baselines=baselines_grad_shap)
        attr_grad_shap = explainer_grad_shap(input_embeds, target=pred, n_samples=100, additional_forward_args=inputs[1].unsqueeze(1))[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_gbprop = GuidedBackpropAttribution(model)
            attr_gbprop = explainer_gbprop(input_embeds, target=pred, additional_forward_args=inputs[1].unsqueeze(1))[0]


        # Compute metrics
        attr_gt = torch.as_tensor(xai_label[0]).unsqueeze(0)

        all_preds.append( pred_base )
        all_labels.append( label )
        for attr, method_name in zip([ attr_our, attr_lig, attr_saliency, attr_dl, attr_dlshap, attr_grad_shap, attr_gbprop ], methods):
            pred_scores = _tokens_to_scores(model, tokens, attr.detach().cpu()).unsqueeze(0)
            metrics_accum[method_name]["iou"].update(pred_scores > threshold, attr_gt)

    return {
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "attribution": {
            method: {
                "iou": float(metrics_accum[method]["iou"].compute())
            }
            for method in metrics_accum
        }
    }


def eval_image(model, ds_train, ds_test, threshold=0.5):
    methods = [ "ours", "layer-ig", "saliency", "deeplift", "deeplift-shap", "gradient-shap", "guided-backprop" ]
    metrics_accum = {
        method: {
            "iou": MulticlassJaccardIndex(2, average="macro"),
        } for method in methods
    }
    all_preds = []
    all_labels = []

    base_model = get_model_wrapper(model) # Model without explainer

    for i in tqdm(range(len(ds_test))):
        in_image, label, xai_label = ds_test[i]

        with torch.no_grad():
            inputs = model.preprocess(in_image.unsqueeze(0))
            logits, _ = model(inputs)
            logits_base = base_model(inputs["pixel_values"])
        pred = torch.argmax(logits).item()
        pred_base = torch.argmax(logits_base).item()

        # Prepare input for explainers
        inputs = (inputs["pixel_values"], )

        # Our method
        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0].mean(dim=1, keepdim=True)

        # Other baselines
        explainer_lig = LayerIntegratedGradientsAttribution(model, baselines=(inputs[0] * 0))
        attr_lig = explainer_lig(inputs[0], target=pred)[0].mean(dim=1, keepdim=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_saliency = SaliencyAttribution(model)
            attr_saliency = explainer_saliency(inputs, target=pred)[0].mean(dim=1, keepdim=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dl = DeepLiftAttribution(model, baselines=(inputs[0] * 0.0))
            attr_dl = explainer_dl(inputs, target=pred)[0].mean(dim=1, keepdim=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baselines_dlshap = torch.cat([ model.preprocess(ds_train[i][0])["pixel_values"] for i in range(20) ], dim=0)
            explainer_dlshap = DeepLiftShapAttribution(model, baselines=baselines_dlshap)
            attr_dlshap = explainer_dlshap(inputs, target=pred)[0].mean(dim=1, keepdim=True)

        baselines_grad_shap = torch.cat([ model.preprocess(ds_train[i][0])["pixel_values"] for i in range(20) ], dim=0)
        explainer_grad_shap = GradientShapAttribution(model, baselines=baselines_grad_shap)
        attr_grad_shap = explainer_grad_shap(inputs[0], target=pred, n_samples=100)[0].mean(dim=1, keepdim=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_gbprop = GuidedBackpropAttribution(model)
            attr_gbprop = explainer_gbprop(inputs, target=pred)[0].mean(dim=1, keepdim=True)


        # Compute metrics
        attr_gt = torch.as_tensor(xai_label).unsqueeze(0)
        all_preds.append( pred_base )
        all_labels.append( label )
        for attr, method_name in zip([ attr_our, attr_lig, attr_saliency, attr_dl, attr_dlshap, attr_grad_shap, attr_gbprop ], methods):
            attr = torchvision.transforms.functional.resize(attr[0].cpu().detach(), (xai_label.shape[1], xai_label.shape[2]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            attr = (attr - attr.min()) / (attr.max() - attr.min())
            metrics_accum[method_name]["iou"].update(attr.unsqueeze(0) > threshold, attr_gt)

    return {
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "attribution": {
            method: {
                "iou": float(metrics_accum[method]["iou"].compute())
            }
            for method in metrics_accum
        }
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Supervised evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to model config directory")
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    L.seed_everything(42)
    model_config = json.load( open(os.path.join(args.config, "config.json"), "r") )
    model_metadata = json.load( open(os.path.join(args.config, "masker/_metadata.json"), "r") )

    ds_train, ds_test, num_classes = get_dataset(
        model_config["dataset_name"], 
        data_dir = "../_datasets", 
        splits = ["train", "test"],
        **model_config["dataset_args"], 
        return_xai_label = True
    )

    model = get_model(
        model_name = model_config["model_name"],
        checkpoint_path = os.path.join(args.config, model_metadata["best_checkpoint"]),
        **model_config["model_args"]
    ).eval().to(args.device)


    match model_config["dataset_name"]:
        case "hatexplain":
            metrics = eval_text(model, ds_train, ds_test, threshold=0.5)
        case "oxford-pet":
            metrics = eval_image(model, ds_train, ds_test, threshold=0.5)
        case _:
            raise NotImplementedError()

    with open(os.path.join(args.config, "metrics-supervised.json"), "w") as f:
        json.dump(metrics, f, indent=4)

