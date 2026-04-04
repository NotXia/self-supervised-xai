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
    'inc-conf': "Inc. conf. $\\uparrow$", 
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
        "\\def\\arraystretch{0.75} \n"
        "\\small \n"
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



def to_latex(config, out_path):
    with open(os.path.join(config, "metrics.json"), "r") as f:
        data = json.load(f)
    
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
        ("../configs/mnist/base", "mnist.tex"),
        ("../configs/cifar10/base", "cifar10.tex"),
        ("../configs/imagenette/base", "imagenette.tex"),
        ("../configs/oxford-pet/base", "oxford-pet.tex"),
        ("../configs/tweet-sentiment/base", "tweet-sentiment.tex"),
        ("../configs/imdb/base", "imdb.tex"),
        ("../configs/politifact/base", "politifact.tex"),
        ("../configs/hatexplain/base", "hatexplain.tex"),
        ("../configs/flickr8k/base", "flickr8k.tex"),
        ("../configs/hateful-memes/base", "hateful-memes.tex"),
        ("../configs/snli-ve/base", "snli-ve.tex"),
        ("../configs/tut-urban/base", "tut-urban.tex"),
        ("../configs/luma/base", "luma.tex"),
        ("../configs/syntheory/base", "syntheory.tex"),
    ]

    for config, out_name in configs:
        to_latex(config, os.path.join(args.out_dir, out_name))