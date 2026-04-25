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

_dataset_name = {
    "../configs/oxford-pet/base": "Oxford-IIIT Pet",
    "../configs/hatexplain/base": "HateXplain",
    "../configs/luma/base": "LUMA",
}


def table2latex(table):
    out = (
        "\\makebox[\\linewidth][c]{% \n"
        "\t\\begin{tabular}{lccc} \n"
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
    table = [
        [ "\\textbf{Method}", *[f"\\textbf{{{_dataset_name[c[0]]}}}" for c in configs] ]
    ]
    baselines = list(_baseline_name.keys())

    data = np.zeros((len(baselines), len(configs)))
    data_std = np.zeros((len(baselines), len(configs)))
    for i, config_runs in enumerate(configs):
        values = {}

        for config in config_runs:
            with open(os.path.join(config, "metrics-supervised.json"), "r") as f:
                metrics = json.load(f)

                for method in metrics["attribution"]:
                    if method not in values: values[method] = []
                    values[method].append( metrics["attribution"][method]["iou"] )
    
        data[:, i] = np.array([ np.mean(values[b]) for b in baselines ])
        data_std[:, i] = np.array([ np.std(values[b]) for b in baselines ])

    
    for i, b in enumerate(baselines):
        row = [ _baseline_name[b] ]
        for j, c in enumerate(configs):
            best_str = f"{max(data[:, j]):.2f}"
            value_str = f"{data[i, j]:.2f}"
            std_str = f"{data_std[i, j]:.2f}"

            if value_str == best_str:
                row.append( f"\\textbf{{{value_str} $\\pm$ {std_str}}}" )
            else:
                row.append( f"{value_str} $\\pm$ {std_str}" )
        table.append(row)

    with open(out_path, "w") as f:
        f.write(table2latex(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LaTex supervised metrics tables creation")
    parser.add_argument("--out-dir", type=str, default="../_tables-cls", help="Output directory")
    args = parser.parse_args()

    os.makedirs( args.out_dir, exist_ok=True )
    
    configs = [
        [ "../configs/oxford-pet/base", "../configs/oxford-pet/base9", "../configs/oxford-pet/base24" ],
        [ "../configs/hatexplain/base", "../configs/hatexplain/base9", "../configs/hatexplain/base24" ],
        [ "../configs/luma/base", "../configs/luma/base9", "../configs/luma/base24" ],
    ]

    to_latex(configs, os.path.join(args.out_dir, "supervised.tex"))