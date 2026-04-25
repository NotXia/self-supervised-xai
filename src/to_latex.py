import argparse
import os
from pathlib import Path
import json
import numpy as np


_baseline_name = {
    'ours': "Ours", 
    'saliency': "Saliency", 
    'guided-backprop': "Guided Backprop.",
    'layer-ig': "Int. Gradients", 
    'deeplift': "DeepLIFT", 
    'deeplift-shap': "DeepLIFT-SHAP", 
    'gradient-shap': "Gradient-SHAP", 
}

_metric_name = {
    'avg-drop': "Avg. drop $\\downarrow$", 
    'inc-conf': "Avg. gain $\\uparrow$", 
    'delete-auc': "Delete AUC $\\downarrow$", 
    'insert-auc': "Insert AUC $\\uparrow$", 
    'complexity': "Complexity $\\downarrow$", 
    'sparsity': "Sparsity $\\uparrow$"
}

_metric_best = {
    'avg-drop': min, 
    'inc-conf': max, 
    'delete-auc': min, 
    'insert-auc': max, 
    'complexity': min, 
    'sparsity': max
}



def table2latex(table):
    out = (
        "\\makebox[\\linewidth][c]{% \n"
        "\t\\begin{tabular}{lcccccc} \n"
        "\t\t\\toprule \n"
        "\t\t" + " & ".join(table[0]) + " \\\\ \n"
        "\t\t\\midrule \n"
    )
    for row in table[1:]:
        out += "\t\t" + " & ".join(row) + " \\\\ \n"
    out += (
        "\t\t\\bottomrule \n"
        "\t\\end{tabular} \n"
        "}"
    )
    return out



def to_latex(configs, out_path):
    # with open(os.path.join(config, "metrics.json"), "r") as f:
    #     data = json.load(f)

    data = {}

    # Load each run
    for config in configs:
        with open(os.path.join(config, "metrics.json"), "r") as f:
            m = json.load(f)
            for method in m:
                if method not in data: data[method] = {}
                for metric_name in m[method]:
                    if metric_name not in data[method]: data[method][metric_name] = []
                    data[method][metric_name].append( m[method][metric_name]["mean"] )

    # Average
    for method in data:
        for metric_name in data[method]:
            values = data[method][metric_name]
            data[method][metric_name] = {}
            data[method][metric_name]["mean"] = np.mean(values)
            data[method][metric_name]["std"] = np.std(values)
    
    baselines = list(_baseline_name.keys())
    metrics = list(_metric_name.keys())

    data_mean = np.array([ [ data[b][m]["mean"] for m in metrics ] for b in baselines ])
    data_std = np.array([ [ data[b][m]["std"] for m in metrics ] for b in baselines ])

    table = [
        [ "\\textbf{Method}", *[f"\\textbf{{{_metric_name[m]}}}" for m in metrics] ]
    ]
    for i, b in enumerate(baselines):
        row = [ _baseline_name[b] ]
        for j, m in enumerate(metrics):
            best_str = f"{_metric_best[m](data_mean[:, j]):.2f}"
            value_mean_str = f"{data_mean[i, j]:.2f}"
            value_std_str = f"{data_std[i, j]:.2f}"

            if value_mean_str == best_str:
                row.append( f"\\textbf{{{value_mean_str} \\std{{{value_std_str}}}}}" )
            else:
                row.append( f"{value_mean_str} \\std{{{value_std_str}}}" )
        table.append(row)

    with open(out_path, "w") as f:
        f.write(table2latex(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LaTex xai metrics tables creation")
    parser.add_argument("--out-dir", type=str, default="../_tables-xai", help="Output directory")
    args = parser.parse_args()

    os.makedirs( args.out_dir, exist_ok=True )
    
    configs = [
        ([ "../configs/mnist/base", "../configs/mnist/base9", "../configs/mnist/base24"], "mnist.tex"),
        ([ "../configs/cifar10/base", "../configs/cifar10/base9", "../configs/cifar10/base24"], "cifar10.tex"),
        ([ "../configs/imagenette/base", "../configs/imagenette/base9", "../configs/imagenette/base24"], "imagenette.tex"),
        # ("../configs/oxford-pet/base", "oxford-pet.tex"),
        ([ "../configs/tweet-sentiment/base", "../configs/tweet-sentiment/base9", "../configs/tweet-sentiment/base24" ], "tweet-sentiment.tex"),
        ([ "../configs/imdb/base", "../configs/imdb/base9", "../configs/imdb/base24" ], "imdb.tex"),
        ([ "../configs/politifact/base", "../configs/politifact/base9", "../configs/politifact/base24" ], "politifact.tex"),
        # ("../configs/hatexplain/base", "hatexplain.tex"),
        ([ "../configs/flickr8k/base", "../configs/flickr8k/base9", "../configs/flickr8k/base24" ], "flickr8k.tex"),
        ([ "../configs/hateful-memes/base", "../configs/hateful-memes/base9", "../configs/hateful-memes/base24" ], "hateful-memes.tex"),
        ([ "../configs/snli-ve/base", "../configs/snli-ve/base9", "../configs/snli-ve/base24" ], "snli-ve.tex"),
        ([ "../configs/tut-urban/base", "../configs/tut-urban/base9", "../configs/tut-urban/base24" ], "tut-urban.tex"),
        ([ "../configs/luma/base", "../configs/luma/base9", "../configs/luma/base24" ], "luma.tex"),
        ([ "../configs/syntheory/base", "../configs/syntheory/base9", "../configs/syntheory/base24" ], "syntheory.tex"),
    ]

    for config, out_name in configs:
        to_latex(config, os.path.join(args.out_dir, out_name))