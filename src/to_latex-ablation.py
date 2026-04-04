import argparse
import os
from pathlib import Path
import json
import numpy as np


_ablation_name = {
    'relu': "No ReLU", 
    'sigmoid': "No sigmoid", 
    'rescale': "No rescale", 
    'loss-binary': "No binary pen.", 
    'loss-magnitude': "No magnitude pen.", 
    'no-loss': "No penalty",
    'nothing': "Nothing", 
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
        "\t\t" + " & ".join(table[1]) + " \\\\ \n"
        "\t\t\\midrule \n"
    )
    for row in table[2:]:
        out += "\t\t" + " & ".join(row) + " \\\\ \n"
    out += (
        "\t\t\\bottomrule \n"
        "\t\\end{tabular} \n"
        "}"
    )
    return out



def to_latex(config, out_path):
    ablations = list(_ablation_name.keys())
    metrics = list(_metric_name.keys())

    table = [
        [ "\\textbf{Method}", *[f"\\textbf{{{_metric_name[m]}}}" for m in metrics] ]
    ]

    with open(os.path.join(config, "base", "metrics.json"), "r") as f:
        data = json.load(f)
    data_mean_full_str = [ f'{data["ours"][m]["mean"]:.2f}' for m in metrics ]
    data_std_full_str = [ f'{data["ours"][m]["std"]:.2f}' for m in metrics ]
    table.append( ["Full"] + [f"{m} \\std{{{s}}}" for m, s in zip(data_mean_full_str, data_std_full_str)] )

    for ablation in ablations:
        with open(os.path.join(config, "ablation", ablation, "metrics.json"), "r") as f:
            data = json.load(f)
        data_mean_full_str = [ f'{data["ours"][m]["mean"]:.2f}' for m in metrics ]
        data_std_full_str = [ f'{data["ours"][m]["std"]:.2f}' for m in metrics ]
        table.append( [_ablation_name[ablation]] + [f"{m} \\std{{{s}}}" for m, s in zip(data_mean_full_str, data_std_full_str)] )

    with open(out_path, "w") as f:
        f.write(table2latex(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LaTex xai ablation metrics tables creation")
    parser.add_argument("--out-dir", type=str, default="../_tables-ablation", help="Output directory")
    args = parser.parse_args()

    os.makedirs( args.out_dir, exist_ok=True )
    
    configs = [
        ("../configs/imagenette", "imagenette.tex"),
        ("../configs/tweet-sentiment", "tweet-sentiment.tex"),
        ("../configs/tut-urban", "tut-urban.tex"),
    ]

    for config, out_name in configs:
        to_latex(config, os.path.join(args.out_dir, out_name))