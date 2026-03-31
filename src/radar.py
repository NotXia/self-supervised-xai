import argparse
import os
from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True



def plot_radar(config, out_path):
    with open(os.path.join(config, "metrics.json"), "r") as f:
        metrics = json.load(f)
        data = np.array([
            [  metrics[k]["inc-conf"]["mean"], metrics[k]["avg-drop"]["mean"], metrics[k]["insert-auc"]["mean"], metrics[k]["delete-auc"]["mean"], metrics[k]["complexity"]["mean"], metrics[k]["sparsity"]["mean"] ]
            for k in metrics.keys()
        ])

    colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink" ]
    labels = ["Increase in confidence", "Average drop (inv.)",  "Insertion AUC", "Deletion AUC (inv.)", "Complexity (inv.)", "Compactness"]
    methods = ["Ours", "Int. Gradients", "Saliency", "DeepLIFT", "DeepLIFT-SHAP", "Gradient-SHAP", "Guided Backprop."]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]


    axis_config = [
        { "min": data[:, 0].min(), "max": data[:, 0].max(), "ticks": np.round(np.linspace(data[:, 0].min(), data[:, 0].max(), 4)[1:-1], 2) }, 
        { "min": data[:, 1].min(), "max": data[:, 1].max(), "ticks": np.round(np.linspace(data[:, 1].min(), data[:, 1].max(), 4)[1:-1], 2) }, 
        { "min": data[:, 2].min(), "max": data[:, 2].max(), "ticks": np.round(np.linspace(data[:, 2].min(), data[:, 2].max(), 4)[1:-1], 2) }, 
        { "min": data[:, 3].min(), "max": data[:, 3].max(), "ticks": np.round(np.linspace(data[:, 3].min(), data[:, 3].max(), 4)[1:-1], 2) }, 
        { "min": data[:, 4].min(), "max": data[:, 4].max(), "ticks": np.round(np.linspace(data[:, 4].min(), data[:, 4].max(), 4)[1:-1]).astype(int) }, 
        { "min": data[:, 5].min(), "max": data[:, 5].max(), "ticks": np.round(np.linspace(data[:, 5].min(), data[:, 5].max(), 4)[1:-1]).astype(int) }
    ]

    invert_idxs = [1, 3, 4]

    for i in range(6):
        data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())
    for i in invert_idxs:
        data[:, i] = 1 - (data[:, i] / np.max(data[:, i]))


    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    for i, (angle, cfg) in enumerate(zip(angles, axis_config)):
        for j, tick in enumerate(cfg['ticks']):
            norm_tick = (tick - cfg['min']) / (cfg['max'] - cfg['min'])
            if i in invert_idxs:
                label = f"{cfg['ticks'][-j-1]:,}"
            else:
                label = f"{tick:,}"
            ax.text(angle, norm_tick, label,
                    ha='center', va='center',
                    fontsize=14, color='black',
                    bbox= { "boxstyle": "round,pad=0.1", "fc": "none", "ec": "none" })

    for values, color, method in zip(data[::-1], colors[::-1], methods[::-1]):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", color=color, linewidth=2, label=method, alpha=1)
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for i, (angle, label) in enumerate(zip(angles[:-1], labels)):
        rotation = [270, -30, 30, 90, 150, 210][i]
        ax.text(
            angle, 1.15,
            label,
            ha='center',
            va='center',
            rotation=rotation,
        )
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Radar plot creation")
    parser.add_argument("--config", type=str, required=True, help="Path to model config directory")
    parser.add_argument("--out", type=str, required=True, help="Output path")
    args = parser.parse_args()

    os.makedirs( Path(args.out).parent, exist_ok=True )
    plot_radar(args.config, args.out)