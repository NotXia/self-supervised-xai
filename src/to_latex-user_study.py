import argparse
import os
import pandas as pd
import numpy as np
from tabulate import tabulate


modalities = ["image", "text", "audio", "multimodal"]
methods = ["ours", "saliency", "int_gradients", "deeplift", "gradient_shap"]

modality2name = {
    "image": "Image",
    "text": "Text",
    "audio": "Audio",
    "multimodal": "Multimodal",
}

method2name = {
    "ours": "Ours",
    "saliency": "Saliency",
    "int_gradients": "Int. Gradients",
    "deeplift": "DeepLIFT",
    "gradient_shap": "Gradient-SHAP",
}


def table2latex(table):
    out = (
        "\\makebox[\\linewidth][c]{% \n"
        "\t\\begin{tabular}{lcccc} \n"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LaTex supervised metrics tables creation")
    parser.add_argument("--out-dir", type=str, default="../_tables-user_study", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv("../samples-user-study/results.csv")

    out_means = np.zeros((len(methods), len(modalities)))
    out_stds = np.zeros((len(methods), len(modalities)))

    for i, modality in enumerate(modalities):
        for j, method in enumerate(methods):
            cols = [c for c in df.columns if c.startswith(modality) and c.endswith(method)]
            per_participant_mean = df[cols].mean()
            out_means[j, i] = np.mean( per_participant_mean )
            out_stds[j, i] = np.std( per_participant_mean )

    table = [
        [ "\\textbf{Modality}", *[f"\\textbf{{{modality2name[m]}}}" for m in modalities] ]
    ]
    for i, method in enumerate(methods):
        row = [ f"{method2name[method]}"]
        for j in range(len(modalities)):
            max_str = f"{np.max(out_means[:, j]):.2f}"
            mean_str = f"{out_means[i, j]:.2f}"
            std_str = f"{out_stds[i, j]:.2f}"
            if max_str == mean_str:
                row.append( f"\\textbf{{{mean_str} \\std{{{std_str}}}}}")
            else:
                row.append( f"{mean_str} \\std{{{std_str}}}")
        table.append(row)

    with open(os.path.join(args.out_dir, "user_study.tex"), "w") as f:
        f.write(table2latex(table))