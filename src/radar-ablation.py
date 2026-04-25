import argparse
import os
from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


_ablation_name = {
    "full": "Full",
    "relu": "No ReLU", 
    "sigmoid": "No sigmoid", 
    "rescale": "No rescale", 
    "loss-binary": "No binary pen.", 
    "loss-magnitude": "No magnitude pen.", 
    "no-loss": "No penalty",
    "nothing": "Nothing", 
}


def plot_radar(config, out_path):
    ablations = list(_ablation_name.keys())
    colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray" ]
    labels = ["Avg. gain", "Avg. drop\n(inv.)",  "Ins. AUC", "Del. AUC\n(inv.)", "Complexity\n(inv.)", "Sparsity"]

    data = []
    with open(os.path.join(config, "base/metrics.json"), "r") as f:
        metrics = json.load(f)
        data.append([
            metrics["ours"]["inc-conf"]["mean"], metrics["ours"]["avg-drop"]["mean"], metrics["ours"]["insert-auc"]["mean"], metrics["ours"]["delete-auc"]["mean"], metrics["ours"]["complexity"]["mean"], metrics["ours"]["sparsity"]["mean"]
        ])
    for ablation in ablations:
        if ablation == "full": continue
        with open(os.path.join(config, "ablation", ablation, "metrics.json"), "r") as f:
            metrics = json.load(f)
            data.append([
                metrics["ours"]["inc-conf"]["mean"], metrics["ours"]["avg-drop"]["mean"], metrics["ours"]["insert-auc"]["mean"], metrics["ours"]["delete-auc"]["mean"], metrics["ours"]["complexity"]["mean"], metrics["ours"]["sparsity"]["mean"]
            ])
    data = np.array(data)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    axis_config = [
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": 0.0, "max": 1.0, "ticks": np.round(np.linspace(0.0, 1.0, 4)[1:-1], 1) }, 
        { "min": data[:, 4].min(), "max": data[:, 4].max(), "ticks": np.round(np.linspace(data[:, 4].min(), data[:, 4].max(), 4)[1:-1]).astype(int) }, 
        { "min": data[:, 5].min(), "max": data[:, 5].max(), "ticks": np.round(np.linspace(data[:, 5].min(), data[:, 5].max(), 4)[1:-1]).astype(int) }
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

    for values, color, ablation in zip(data, colors, ablations):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", color=color, linewidth=2, label=_ablation_name[ablation], alpha=1, zorder=2 if ablation == "full" else 1)
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
    parser.add_argument("--out-dir", type=str, default="../_radar-ablation", help="Output path")
    args = parser.parse_args()

    os.makedirs( args.out_dir, exist_ok=True )

    configs = [
        ("../configs/imagenette", "imagenette.pdf"),
        ("../configs/imdb", "imdb.pdf"),
        ("../configs/tut-urban", "tut-urban.pdf"),
    ]

    for config, out_name in configs:
        plot_radar(config, os.path.join(args.out_dir, out_name))