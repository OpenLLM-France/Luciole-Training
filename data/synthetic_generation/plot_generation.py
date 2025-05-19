from datasets import load_from_disk
import re
from collections import Counter
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_educational_json(text: str) -> dict | None:
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)

    matches = pattern.findall(text)
    match = matches[0] 
    try:
        data_dict = json.loads(match)
        return data_dict
    except json.JSONDecodeError:
        return None

def plot_label_crosstab(ds, col1_name, col2_name, expe_path, output_name):
    """
    input_1: tuple (name, list/array) for first column (e.g. predicted)
    input_2: tuple (name, list/array) for second column (e.g. true labels)
    expe_path: folder path to save the plot
    output_name: file name (without extension)
    """

    # Create DataFrame
    if (col1_name not in ds.column_names) or (col2_name not in ds.column_names):
        return None
    df = pd.DataFrame({col2_name: ds[col2_name], col1_name: ds[col1_name]})

    # Cross-tabulation (contingency table)
    cross_tab = pd.crosstab(df[col2_name], df[col1_name])

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Cross-tabulation between {col2_name} and {col1_name}")
    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.join(expe_path, "out"), exist_ok=True)
    plt.savefig(os.path.join(expe_path, "out", f"crosstab_{output_name}.png"))
    plt.close()


def plot_histograms(ds, expe_path):
    """
    Plots histograms or bar charts for the specified metrics in the dataset.

    Args:
        ds: A dataset (e.g., pandas DataFrame or Hugging Face Dataset).
        expe_path: Path to save the output plot.
        metrics: List of fields to plot. Numeric fields are plotted as histograms,
                 categorical and boolean fields as bar charts.
    """
    metrics = ['educational_score', 'toxicity_score', 'topic', 'is_ad', 'is_toxic']
    n_plots = len(metrics)
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
    if n_plots == 1:
        axs = [axs]  # Ensure axs is always iterable

    for i, metric in enumerate(metrics):
        if metric not in ds.column_names:
            axs[i].text(0.5, 0.5, f'{metric} not found', ha='center', va='center')
            axs[i].set_title(f'{metric} (missing)')
            axs[i].axis('off')
            continue

        values = ds[metric]

        # Handle missing or None values
        values = [v for v in values if v is not None]

        # Determine type
        if all(isinstance(v, bool) for v in values):
            counts = Counter(values)
            axs[i].bar(['False', 'True'], [counts.get(False, 0), counts.get(True, 0)], alpha=0.7)
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')

        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            axs[i].hist(values, bins=6, range=(0, 6), alpha=0.7, align='left', rwidth=0.8)
            axs[i].set_xlabel('Score')
            axs[i].set_ylabel('Frequency')

        else:
            counts = Counter(values)
            axs[i].bar(counts.keys(), counts.values(), alpha=0.7)
            axs[i].set_xlabel('Category')
            axs[i].set_ylabel('Frequency')
            axs[i].tick_params(axis='x', rotation=90)

        axs[i].set_title(metric.replace('_', ' ').title())

    plt.tight_layout()
    os.makedirs(os.path.join(expe_path, "out"), exist_ok=True)
    plt.savefig(os.path.join(expe_path, "out", "hist.png"))
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--expe_path",
        type=str,
    )
    args = argparser.parse_args()
    expe_path = args.expe_path

    ds = load_from_disk(os.path.join(expe_path, "default"))['train']
    ds = ds.map(lambda x: extract_educational_json(x["generation"]))
    print(ds[0])

    plot_label_crosstab(ds, 'educational_score', 'toxicity_score', expe_path, "toxic_edu")
    plot_label_crosstab(ds, 'educational_score', 'is_toxic', expe_path, "toxic_edu")
    plot_label_crosstab(ds, 'educational_score', 'is_ad', expe_path, "ad_edu")
    plot_label_crosstab(ds, 'educational_score', 'topic', expe_path, "edu_topic")
    plot_label_crosstab(ds, 'is_toxic', 'topic', expe_path, "toxic_topic")
    plot_label_crosstab(ds, 'toxicity_score', 'topic', expe_path, "toxic_topic")

    plot_histograms(ds, expe_path)

