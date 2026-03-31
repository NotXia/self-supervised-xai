import argparse
import os
import json
import random
import itertools
import warnings
from tqdm.auto import tqdm
from tabulate import tabulate
import gc

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import lightning as L

import numpy as np
import matplotlib.pyplot as plt

from models import get_model
from data import get_dataset
from utils.plot import *
from xai import *
from metrics import *

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True



def _accum_metrics(metrics, model_base, inputs, attribution):
    metrics["avg-drop"].accumulate(model_base, inputs, attribution)
    metrics["inc-conf"].accumulate(model_base, inputs, attribution)
    metrics["delete-auc"].accumulate(model_base, inputs, attribution)
    metrics["insert-auc"].accumulate(model_base, inputs, attribution)
    metrics["complexity"].accumulate(attribution)
    metrics["sparsity"].accumulate(attribution)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU to use")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    L.seed_everything(args.seed)

    model_config = json.load( open(os.path.join(args.config, "config.json"), "r") )
    model_metadata = json.load( open(os.path.join(args.config, "masker/_metadata.json"), "r") )
    device = args.device 

    # Load dataset
    ds_train, ds_val, ds_test, num_classes = get_dataset(
        model_config["dataset_name"], 
        data_dir = "../_datasets", 
        splits = ["train", "validation", "test"],
        **model_config["dataset_args"], 
    )

    # Load model
    model = get_model(
        model_name = model_config["model_name"],
        checkpoint_path = os.path.join(args.config, model_metadata["best_checkpoint"]),
        **model_config["model_args"]
    ).eval().to(device)
    model_base = get_model_wrapper(model) # Model without explainer module


    methods = [ "ours", "layer-ig", "saliency", "deeplift", "deeplift-shap", "gradient-shap", "guided-backprop" ]
    metrics = {
        method: {
            "avg-drop": AverageDropMetric(device),
            "inc-conf": IncreaseConfidenceMetric(device),
            "delete-auc": DeletionAUCMetric(device),
            "insert-auc": InsertionAUCMetric(device),
            "complexity": ComplexityMetric(device),
            "sparsity": SparsityMetric(device)
        } for method in methods
    }


    for i in tqdm(range(len(ds_test))):
        text, label = ds_test[i]

        with torch.no_grad():
            inputs = model.preprocess(text)
            logits, weights = model(inputs)
        pred = torch.argmax(logits).item()

        inputs = (inputs["input_ids"], inputs["attention_mask"])
        input_embeds = model.text_encoder.embed(inputs[0])


        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0]
        _accum_metrics(metrics["ours"], model_base, inputs, attr_our)

        explainer_lig = IntegratedGradientsAttribution(model, baselines=(input_embeds * 0))
        attr_lig = explainer_lig(input_embeds, target=pred, additional_forward_args=inputs[1])[0][0]
        _accum_metrics(metrics["layer-ig"], model_base, inputs, attr_lig)

        explainer_saliency = SaliencyAttribution(model)
        attr_saliency = explainer_saliency(input_embeds, target=pred, additional_forward_args=inputs[1])[0]
        _accum_metrics(metrics["saliency"], model_base, inputs, attr_saliency)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dl = DeepLiftAttribution(model, baselines=(input_embeds * 0.0))
            attr_dl = explainer_dl(input_embeds, target=pred, additional_forward_args=inputs[1])[0]
            _accum_metrics(metrics["deeplift"], model_base, inputs, attr_dl)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baselines_dlshap = torch.cat([ model.text_encoder.embed(model.preprocess(ds_train[i][0])["input_ids"]) for i in range(20) ], dim=0)
            explainer_dlshap = DeepLiftShapAttribution(model, baselines=baselines_dlshap)
            attr_dlshap = explainer_dlshap(input_embeds, target=pred, additional_forward_args=inputs[1].unsqueeze(1))[0]
            _accum_metrics(metrics["deeplift-shap"], model_base, inputs, attr_dlshap)

        baselines_grad_shap = torch.cat([ model.text_encoder.embed(model.preprocess(ds_train[i][0])["input_ids"]) for i in range(20) ], dim=0)
        explainer_grad_shap = GradientShapAttribution(model, baselines=baselines_grad_shap)
        attr_grad_shap = explainer_grad_shap(input_embeds, target=pred, n_samples=100, additional_forward_args=inputs[1].unsqueeze(1))[0]
        _accum_metrics(metrics["gradient-shap"], model_base, inputs, attr_grad_shap)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_gbprop = GuidedBackpropAttribution(model)
            attr_gbprop = explainer_gbprop(input_embeds, target=pred, additional_forward_args=inputs[1].unsqueeze(1))[0]
            _accum_metrics(metrics["guided-backprop"], model_base, inputs, attr_gbprop)

        torch.cuda.empty_cache()
        gc.collect()

    # Print results
    tab_out = []
    for method in metrics:
        out = [ method ]
        for metric in metrics[method]:
            mean, std = metrics[method][metric].compute()
            out.append( f"{mean:.2f} +/- {std:.2f}" )
        tab_out.append( out )

    print(tabulate(
        tab_out, 
        headers = ["method", *metrics["ours"]]
    ))

    # Save results
    json_out = {}
    for method in metrics:
        json_out[method] = {}
        for metric in metrics[method]:
            mean, std = metrics[method][metric].compute()
            json_out[method][metric] = { "mean": mean, "std": std }
    json.dump(json_out, open(os.path.join(args.config, "metrics.json"), "w"), indent=4)