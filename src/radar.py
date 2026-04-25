import argparse
import os
from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import defaultdict

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True



def plot_radar(configs, out_path):
    metrics = {}

    # Load each run
    for config in configs:
        with open(os.path.join(config, "metrics.json"), "r") as f:
            m = json.load(f)
            for method in m:
                if method not in metrics: metrics[method] = {}
                for metric_name in m[method]:
                    if metric_name not in metrics[method]: metrics[method][metric_name] = 0.0
                    metrics[method][metric_name] += m[method][metric_name]["mean"]

    # Average
    for method in metrics:
        for metric_name in metrics[method]:
            metrics[method][metric_name] = metrics[method][metric_name] / len(configs)

    data = np.array([
        [  metrics[k]["inc-conf"], metrics[k]["avg-drop"], metrics[k]["insert-auc"], metrics[k]["delete-auc"], metrics[k]["complexity"], metrics[k]["sparsity"] ]
        for k in metrics.keys()
    ])

    colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink" ]
    labels = ["Avg. gain", "Avg. drop\n(inv.)",  "Ins. AUC", "Del. AUC\n(inv.)", "Complexity\n(inv.)", "Sparsity"]
    methods = ["Ours", "Int. Gradients", "Saliency", "DeepLIFT", "DeepLIFT-SHAP", "Gradient-SHAP", "Guided Backprop."]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    axis_config = [
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": data[:, 4].max()*1.1, "ticks": np.round(np.linspace(0.0, data[:, 4].max()*1.1, 4)[1:-1]).astype(int) }, 
        { "min": 0.0, "max": data[:, 5].max()*1.1, "ticks": np.round(np.linspace(0.0, data[:, 5].max()*1.1, 4)[1:-1]).astype(int) }
    ]

    invert_idxs = [1, 3, 4]

    for i in range(6):
        data[:, i] = (data[:, i] - axis_config[i]["min"]) / (axis_config[i]["max"] - axis_config[i]["min"])
    for i in invert_idxs:
        data[:, i] = 1 - data[:, i]


    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))

    for i, (angle, cfg) in enumerate(zip(angles, axis_config)):
        for j, tick in enumerate(cfg['ticks']):
            norm_tick = (tick - cfg['min']) / (cfg['max'] - cfg['min'])
            if i in invert_idxs:
                label = f"{cfg['ticks'][-j-1]:,}"
            else:
                label = f"{tick:,}"
            ax.text(angle, norm_tick, label,
                    ha='center', va='center',
                    fontsize=12, color='black',
                    bbox= { "boxstyle": "round,pad=0.1", "fc": "none", "ec": "none" })

    for values, color, method in zip(data, colors, methods):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", color=color, linewidth=2, label=method, alpha=1, zorder=2 if method == "Ours" else 1)
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for i, (angle, label) in enumerate(zip(angles[:-1], labels)):
        rotation = [270, -30, 30, 90, 150, 210][i]
        ax.text(
            angle, 1.3,
            label,
            ha='center',
            va='center',
            rotation=rotation,
        )
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)

    label_params = plt.gca().get_legend_handles_labels() 
    figl, axl = plt.subplots()
    axl.axis(False)
    leg = axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 50}, borderpad=0, frameon=False, ncols=1)
    figl.canvas.draw()
    bbox = leg.get_window_extent().transformed(figl.dpi_scale_trans.inverted())
    figl.savefig(os.path.join(Path(out_path).parent, "legend.pdf"), bbox_inches=bbox, pad_inches=0)
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Radar plot creation")
    parser.add_argument("--out-dir", type=str, default="../_radar", help="Output path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs = [
        ([ "../configs/mnist/base", "../configs/mnist/base9", "../configs/mnist/base24"], "mnist.pdf"),
        ([ "../configs/cifar10/base", "../configs/cifar10/base9", "../configs/cifar10/base24"], "cifar10.pdf"),
        ([ "../configs/imagenette/base", "../configs/imagenette/base9", "../configs/imagenette/base24"], "imagenette.pdf"),
        # ("../configs/oxford-pet/base", "oxford-pet.pdf"),
        ([ "../configs/tweet-sentiment/base", "../configs/tweet-sentiment/base9", "../configs/tweet-sentiment/base24" ], "tweet-sentiment.pdf"),
        ([ "../configs/imdb/base", "../configs/imdb/base9", "../configs/imdb/base24" ], "imdb.pdf"),
        ([ "../configs/politifact/base", "../configs/politifact/base9", "../configs/politifact/base24" ], "politifact.pdf"),
        # ("../configs/hatexplain/base", "hatexplain.pdf"),
        ([ "../configs/flickr8k/base", "../configs/flickr8k/base9", "../configs/flickr8k/base24" ], "flickr8k.pdf"),
        ([ "../configs/hateful-memes/base", "../configs/hateful-memes/base9", "../configs/hateful-memes/base24" ], "hateful-memes.pdf"),
        ([ "../configs/snli-ve/base", "../configs/snli-ve/base9", "../configs/snli-ve/base24" ], "snli-ve.pdf"),
        ([ "../configs/tut-urban/base", "../configs/tut-urban/base9", "../configs/tut-urban/base24" ], "tut-urban.pdf"),
        ([ "../configs/luma/base", "../configs/luma/base9", "../configs/luma/base24" ], "luma.pdf"),
        ([ "../configs/syntheory/base", "../configs/syntheory/base9", "../configs/syntheory/base24" ], "syntheory.pdf"),
    ]

    for configs, out_name in configs:
        plot_radar(configs, os.path.join(args.out_dir, out_name))