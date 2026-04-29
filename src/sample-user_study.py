import argparse
import os
from pathlib import Path
import json
import numpy as np
import warnings

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import lightning as L

from weasyprint import HTML
import fitz 
from PIL import Image, ImageChops
import io
import soundfile as sf

from models import get_model
from data import get_dataset
from utils.plot import *
from xai import *

plt.rcParams["font.size"] = 16



def load(model_dir, device):
    model_config = json.load( open(os.path.join(model_dir, "config.json"), "r") )
    model_metadata = json.load( open(os.path.join(model_dir, "masker/_metadata.json"), "r") )
    
    ds_train, _, ds_test, num_classes = get_dataset(
        model_config["dataset_name"], 
        data_dir = "../_datasets", 
        splits = ["train", "validation", "test"],
        **model_config["dataset_args"], 
    )

    model = get_model(
        model_name = model_config["model_name"],
        checkpoint_path = os.path.join(model_dir, model_metadata["best_checkpoint"]),
        **model_config["model_args"]
    ).eval().to(device)

    return model, ds_train, ds_test


def crop_whitespace(image):
    bg = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image


def sample_text(model, ds_train, ds_test, num_samples, out_dir):
    sampled = 0

    while sampled < num_samples:
        i = np.random.randint(0, len(ds_test))
        text, label = ds_test[i]

        with torch.no_grad():
            inputs = model.preprocess(text)
            logits, weights = model(inputs)
        pred = torch.argmax(logits).item()
        pred_label = ds_test.id2label[pred]

        if pred != label: continue
        if label == 1: continue # Ignore neutral label

        tokens = inputs["input_ids"].cpu().squeeze(0)
        tokens = tokens.tolist()

        inputs = (inputs["input_ids"], inputs["attention_mask"])
        input_embeds = model.text_encoder.embed(inputs[0])
        seq_length = inputs[1].sum().item()

        # Compute attribution maps
        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0]

        explainer_lig = IntegratedGradientsAttribution(model, baselines=(input_embeds * 0))
        attr_lig = explainer_lig(input_embeds, target=pred, additional_forward_args=inputs[1])[0]

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


        # Save attribution maps
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        for i, (attr, name, name_file) in enumerate([ 
            (attr_our, "Ours", "ours"), 
            (attr_saliency, "Saliency", "saliency"), 
            # (attr_gbprop, "Guided Backprop.", "guided_backprop"),
            (attr_lig, "Int. Gradients", "int_gradients"), 
            (attr_dl, "DeepLIFT", "deeplift"), 
            # (attr_dlshap, "DeepLIFT-SHAP", "deeplift_shap"), 
            (attr_grad_shap, "Gradient-SHAP", "gradient_shap"), 
        ]):
            attr_plot = torch.abs(attr)
            attr_plot = attr_plot.mean(dim=2)[0][:seq_length]
            attr_plot = (attr_plot - attr_plot.min()) / (attr_plot.max() - attr_plot.min() + 1e-16)
            attr_plot = attr_plot.tolist()

            text, scores = decode_text_with_scores(tokens[1:], attr_plot[1:], model.text_encoder._model.tokenizer, "</s>")
            html = ' '.join([highlighter_html(t, s) for t, s in zip(text, scores)])

            HTML(string=html).write_pdf(os.path.join(sample_dir, f"{name_file}.pdf"))
            page = fitz.open(os.path.join(sample_dir, f"{name_file}.pdf"))[0]
            os.remove(os.path.join(sample_dir, f"{name_file}.pdf")) 
            image = page.get_pixmap(dpi=300).pil_image()
            image = crop_whitespace(image)
            image.save(os.path.join(sample_dir, f"{name_file}.png"))
        
        html = ' '.join([highlighter_html(t, 0) for t in text])
        HTML(string=html).write_pdf(os.path.join(sample_dir, f"original.pdf"))
        page = fitz.open(os.path.join(sample_dir, f"original.pdf"))[0]
        os.remove(os.path.join(sample_dir, f"original.pdf")) 
        image = page.get_pixmap(dpi=300).pil_image()
        image = crop_whitespace(image)
        image.save(os.path.join(sample_dir, f"original.png"))

        # Save label
        with open(os.path.join(sample_dir, f"label.txt"), "w") as f:
            f.write(pred_label)

        sampled += 1

def sample_image(model, ds_train, ds_test, num_samples, out_dir):
    sampled = 0

    while sampled < num_samples:
        i = np.random.randint(0, len(ds_test))
        in_image, label = ds_test[i]

        with torch.no_grad():
            inputs = model.preprocess(in_image.unsqueeze(0))
            logits, weights = model(inputs)
        pred = torch.argmax(logits).item()
        pred_label = ds_test.id2label[pred]

        if pred != label: continue

        inputs = (inputs["pixel_values"], )

        # Compute attribution maps

        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0].mean(dim=1, keepdim=True)

        explainer_lig = IntegratedGradientsAttribution(model, baselines=(inputs[0] * 0))
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

        # Save attribution maps
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)
        image_plot = in_image.cpu()

        for i, (attr, name, name_file) in enumerate([ 
            (attr_our, "Ours", "ours"), 
            (attr_saliency, "Saliency", "saliency"), 
            # (attr_gbprop, "Guided Backprop.", "guided_backprop"),
            (attr_lig, "Int. Gradients", "int_gradients"), 
            (attr_dl, "DeepLIFT", "deeplift"), 
            # (attr_dlshap, "DeepLIFT-SHAP", "deeplift_shap"), 
            (attr_grad_shap, "Gradient-SHAP", "gradient_shap"), 
        ]):
            attr_plot = torchvision.transforms.functional.resize(attr[0].cpu().detach(), (image_plot.shape[1], image_plot.shape[2]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            attr_plot = torch.abs(attr_plot)
            attr_plot = (attr_plot - attr_plot.min()) / (attr_plot.max() - attr_plot.min())

            im = plt.imshow(image_plot.permute(1, 2, 0))
            plt.imshow(attr_plot.permute(1, 2, 0), alpha=0.75)
            plt.xticks([]); plt.yticks([])
            plt.colorbar(im, ax=plt.gca(), location="right", shrink=0.6, fraction=0.02, pad=0.02)
            plt.savefig(os.path.join(sample_dir, f"{name_file}.png"), bbox_inches="tight")
            plt.close()

        im = plt.imshow(image_plot.permute(1, 2, 0))
        plt.xticks([]); plt.yticks([])
        plt.savefig(os.path.join(sample_dir, f"original.png"), bbox_inches="tight")
        plt.close()

        # Save label
        with open(os.path.join(sample_dir, f"label.txt"), "w") as f:
            f.write(pred_label)

        sampled += 1

def sample_audio(model, ds_train, ds_test, num_samples, out_dir):
    sampled = 0

    while sampled < num_samples:
        i = np.random.randint(0, len(ds_test))
        audio, label = ds_train[i]

        with torch.no_grad():
            inputs = model.preprocess(audio.unsqueeze(0))
            logits, weights = model(inputs)
        pred = torch.argmax(logits).item()
        pred_label = ds_test.id2label[pred]

        if pred != label: continue

        inputs = (inputs["input_values"], )

        # Compute attribution maps
        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)[0]

        explainer_lig = IntegratedGradientsAttribution(model, baselines=(inputs[0] * 0))
        attr_lig = explainer_lig(inputs[0], target=pred)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_saliency = SaliencyAttribution(model)
            attr_saliency = explainer_saliency(inputs, target=pred)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dl = DeepLiftAttribution(model, baselines=(inputs[0] * 0.0))
            attr_dl = explainer_dl(inputs, target=pred)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baselines_dlshap = torch.cat([ model.preprocess(ds_train[i][0])["input_values"].unsqueeze(0) for i in range(20) ], dim=0)
            explainer_dlshap = DeepLiftShapAttribution(model, baselines=baselines_dlshap)
            attr_dlshap = explainer_dlshap(inputs, target=pred)[0]

        baselines_grad_shap = torch.cat([ model.preprocess(ds_train[i][0])["input_values"].unsqueeze(0) for i in range(20) ], dim=0)
        explainer_grad_shap = GradientShapAttribution(model, baselines=baselines_grad_shap)
        attr_grad_shap = explainer_grad_shap(inputs, target=pred, n_samples=50)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_gbprop = GuidedBackpropAttribution(model)
            attr_gbprop = explainer_gbprop(inputs, target=pred)[0]

        # Save attribution maps
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        for i, (attr, name, name_file) in enumerate([ 
            (attr_our, "Ours", "ours"), 
            (attr_saliency, "Saliency", "saliency"), 
            # (attr_gbprop, "Guided Backprop.", "guided_backprop"),
            (attr_lig, "Int. Gradients", "int_gradients"), 
            (attr_dl, "DeepLIFT", "deeplift"), 
            # (attr_dlshap, "DeepLIFT-SHAP", "deeplift_shap"), 
            (attr_grad_shap, "Gradient-SHAP", "gradient_shap"), 
        ]):
            attr = attr[0].cpu().detach()
            attr_mask = torch.abs(attr)
            attr_mask = (attr_mask - attr_mask.min()) / (attr_mask.max() - attr_mask.min() + 1e-16)
            sf.write(os.path.join(sample_dir, f"{name_file}.wav"), audio*attr_mask, 16_000)

        sf.write(os.path.join(sample_dir, f"original.wav"), audio, 16_000)

        # Save label
        with open(os.path.join(sample_dir, f"label.txt"), "w") as f:
            f.write(pred_label)

        sampled += 1

def sample_multimodal(model, ds_train, ds_test, num_samples, out_dir):
    sampled = 0

    while sampled < num_samples:
        idx = np.random.randint(0, len(ds_test))
        image, text, label = ds_test[idx]

        with torch.no_grad():
            image_inputs, text_inputs = model.preprocess([image], [text])
            logits, text_weights, image_weights = model(image_inputs, text_inputs)
        pred = torch.argmax(logits).item()
        pred_label = ds_test.id2label[pred]

        if pred != label: continue

        tokens = text_inputs["input_ids"].cpu().squeeze(0)
        tokens = tokens.tolist()
        inputs = (image_inputs["pixel_values"], text_inputs["input_ids"], text_inputs["attention_mask"])
        text_embeds = model.text_encoder.embed(inputs[1])

        baselines_text = []
        baselines_image = []
        for i in range(20):
            image_inputs, text_inputs = model.preprocess([ds_train[i][0]], [ds_train[i][1]])
            text_embeds = model.text_encoder.embed(text_inputs["input_ids"])
            baselines_text.append(text_embeds)
            baselines_image.append(image_inputs["pixel_values"])
        baselines_text = torch.cat(baselines_text)
        baselines_image = torch.cat(baselines_image)

        # Compute attribution maps
        explainer_our = OurAttribution(model)
        attr_our = explainer_our(inputs)

        explainer_lig = IntegratedGradientsAttribution(model, baselines=(inputs[0]*0, text_embeds*0))
        attr_lig = explainer_lig((inputs[0], text_embeds), target=pred, additional_forward_args=inputs[2])

        explainer_saliency = SaliencyAttribution(model)
        attr_saliency = explainer_saliency((inputs[0], text_embeds), target=pred, additional_forward_args=inputs[2])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dl = DeepLiftAttribution(model, baselines=(inputs[0]*0.0, text_embeds*0.0))
            attr_dl = explainer_dl((inputs[0], text_embeds), target=pred, additional_forward_args=inputs[2])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_dlshap = DeepLiftShapAttribution(model, baselines=(baselines_image, baselines_text))
            attr_dlshap = explainer_dlshap((inputs[0], text_embeds), target=pred, additional_forward_args=inputs[2].unsqueeze(1))

        explainer_grad_shap = GradientShapAttribution(model, baselines=(baselines_image, baselines_text))
        attr_grad_shap = explainer_grad_shap((inputs[0], text_embeds), target=pred, n_samples=100, additional_forward_args=inputs[2])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer_gbprop = GuidedBackpropAttribution(model)
            attr_gbprop = explainer_gbprop((inputs[0], text_embeds), target=pred, additional_forward_args=inputs[2])


        # Save attribution maps
        sample_dir = os.path.join(out_dir, f"sample_{idx}")
        os.makedirs(sample_dir, exist_ok=True)
        image_plot = image.cpu()

        for i, (attr, name, name_file) in enumerate([ 
            (attr_our, "Ours", "ours"), 
            (attr_saliency, "Saliency", "saliency"), 
            # (attr_gbprop, "Guided Backprop.", "guided_backprop"),
            (attr_lig, "Int. Gradients", "int_gradients"), 
            (attr_dl, "DeepLIFT", "deeplift"), 
            # (attr_dlshap, "DeepLIFT-SHAP", "deeplift_shap"), 
            (attr_grad_shap, "Gradient-SHAP", "gradient_shap"), 
        ]):
            attr_image_plot = attr[0].mean(dim=1, keepdim=True)
            attr_image_plot = torchvision.transforms.functional.resize(attr_image_plot[0].cpu().detach(), (image_plot.shape[1], image_plot.shape[2]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            attr_image_plot = torch.abs(attr_image_plot)
            attr_image_plot = (attr_image_plot - attr_image_plot.min()) / (attr_image_plot.max() - attr_image_plot.min())

            attr_text_plot = attr[1].mean(dim=2)[0]
            attr_text_plot = torch.abs(attr_text_plot)
            attr_text_plot = (attr_text_plot - attr_text_plot.min()) / (attr_text_plot.max() - attr_text_plot.min())
            attr_text_plot = attr_text_plot.tolist()

            text, scores = decode_text_with_scores(tokens[1:], attr_text_plot[1:], model.text_encoder._model.tokenizer, "</s>")
            html = ' '.join([highlighter_html(t, s) for t, s in zip(text, scores)])
            HTML(string=html).write_pdf(os.path.join(sample_dir, f"{name_file}.pdf"))
            page = fitz.open(os.path.join(sample_dir, f"{name_file}.pdf"))[0]
            os.remove(os.path.join(sample_dir, f"{name_file}.pdf")) 
            image_txt = page.get_pixmap(dpi=300).pil_image()
            image_txt = crop_whitespace(image_txt)

            im = plt.imshow(image_plot.permute(1, 2, 0))
            plt.imshow(attr_image_plot.permute(1, 2, 0), alpha=0.75)
            plt.xticks([]); plt.yticks([])
            plt.colorbar(im, ax=plt.gca(), location="right", shrink=0.6, fraction=0.02, pad=0.02)
            buff = io.BytesIO()
            plt.savefig(buff, format="png")
            buff.seek(0)
            image_img = Image.open(buff)
            plt.close()

            # Concatenate images
            out_width = max(image_txt.size[0], image_img.size[0])
            out_height = image_txt.size[1] + image_img.size[1]
            image_out = Image.new('RGB', (out_width, out_height), color=(255, 255, 255))
            image_out.paste(image_txt, (0, 0))
            image_out.paste(image_img, (0, image_txt.size[1]))

            image_out.save(os.path.join(sample_dir, f"{name_file}.png"))

        html = ' '.join([highlighter_html(t, 0) for t in text])
        HTML(string=html).write_pdf(os.path.join(sample_dir, f"{name_file}.pdf"))
        page = fitz.open(os.path.join(sample_dir, f"{name_file}.pdf"))[0]
        os.remove(os.path.join(sample_dir, f"{name_file}.pdf")) 
        image_txt = page.get_pixmap(dpi=300).pil_image()
        image_txt = crop_whitespace(image_txt)
        im = plt.imshow(image_plot.permute(1, 2, 0))
        plt.xticks([]); plt.yticks([])
        buff = io.BytesIO()
        plt.savefig(buff, format="png")
        buff.seek(0)
        image_img = Image.open(buff)
        plt.close()
        # Concatenate images
        out_width = max(image_txt.size[0], image_img.size[0])
        out_height = image_txt.size[1] + image_img.size[1]
        image_out = Image.new('RGB', (out_width, out_height), color=(255, 255, 255))
        image_out.paste(image_txt, (0, 0))
        image_out.paste(image_img, (0, image_txt.size[1]))
        image_out.save(os.path.join(sample_dir, f"original.png"))

        # Save label
        with open(os.path.join(sample_dir, f"label.txt"), "w") as f:
            f.write(pred_label)

        sampled += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Sampler for user study")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--out-dir", type=str, default="_study", help="Device to use")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs( args.out_dir, exist_ok=True )
    

    L.seed_everything(args.seed)
    sample_text(*load("../configs/tweet-sentiment/base", args.device), args.num_samples, os.path.join(args.out_dir, "text"))

    L.seed_everything(args.seed)
    sample_image(*load("../configs/imagenette/base", args.device), args.num_samples, os.path.join(args.out_dir, "image"))

    L.seed_everything(args.seed)
    sample_audio(*load("../configs/luma/base", args.device), args.num_samples, os.path.join(args.out_dir, "audio"))

    L.seed_everything(args.seed)
    sample_multimodal(*load("../configs/flickr8k/base", args.device), args.num_samples, os.path.join(args.out_dir, "multimodal"))