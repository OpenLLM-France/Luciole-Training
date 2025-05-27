from datasets import load_from_disk
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import extract_educational_json


def plot_label_crosstab(data, output_name):
    # Create DataFrame
    df = pd.DataFrame(data)
    columns = df.columns

    # Cross-tabulation (contingency table)
    cross_tab = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
    plt.title("Cross-tabulation")
    plt.xlabel(columns[0].split("/")[-1])
    plt.ylabel(columns[1].split("/")[-1])
    plt.tight_layout()

    # Save plot
    plt.savefig(output_name)
    plt.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--expe_1",
        type=str,
    )
    argparser.add_argument(
        "--expe_2",
        type=str,
    )
    argparser.add_argument(
        "--output_name",
        type=str,
        default="out/crosstab.png",
    )
    args = argparser.parse_args()
    expe_1 = args.expe_1
    expe_2 = args.expe_2
    output_name = args.output_name
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    out = {}
    for expe_path in [expe_1, expe_2]:
        ds = load_from_disk(os.path.join(expe_path, "default"))["train"]
        ds = ds.map(lambda x: extract_educational_json(x["generation"]))
        out[expe_path] = ds["educational_score"]

    plot_label_crosstab(out, output_name)
