from datasets import load_from_disk
import re
from collections import Counter
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

categories = [
    "News",
    "Sports",
    "Culture",
    "Entertainment",
    "Technology",
    "Business",
    "Health",
    "Science",
    "Mathematics",
    "Travel",
    "Education",
    "Lifestyle",
    "Politics",
    "Finance",
    "Real Estate",
    "Shopping",
    "Dating",
    "Adult"
]

def extract_educational_json(text: str) -> dict | None:
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)

    matches = pattern.findall(text)
    match = matches[0] 
    try:
        data_dict = json.loads(match)
        return data_dict
    except json.JSONDecodeError:
        return None

def plot_label_crosstab(input_1, input_2, expe_path, output_name):
    """
    input_1: tuple (name, list/array) for first column (e.g. predicted)
    input_2: tuple (name, list/array) for second column (e.g. true labels)
    expe_path: folder path to save the plot
    output_name: file name (without extension)
    """

    col1_name, col1_vals = input_1
    col2_name, col2_vals = input_2

    # Create DataFrame
    df = pd.DataFrame({col2_name: col2_vals, col1_name: col1_vals})

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
    # Extract columns
    educational_scores = ds['educational_score']
    harmfulness_scores = ds['toxicity_score']
    topics = ds['topic']

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Histogram for educational_score
    axs[0].hist(educational_scores, bins=6, range=(0, 6), alpha=0.7, align='left', rwidth=0.8)
    axs[0].set_title('Educational Score')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Frequency')

    # Histogram for harmfulness_score
    axs[1].hist(harmfulness_scores, bins=6, range=(0, 6), alpha=0.7, align='left', rwidth=0.8)
    axs[1].set_title('Toxicity Score')
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Frequency')

    # Histogram for topic (categorical) — count frequencies
    topic_counts = Counter(topics)
    axs[2].bar(topic_counts.keys(), topic_counts.values(), alpha=0.7)
    axs[2].set_title('Topic Distribution')
    axs[2].set_xlabel('Topic')
    axs[2].set_ylabel('Frequency')
    axs[2].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.join(expe_path, "out"), exist_ok=True)
    plt.savefig(os.path.join(expe_path, "out", f"hist.png"))

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
    ds = ds.map(lambda x: {"topic": x["topic"] if x["topic"] in categories else "Other"})
    print(ds[0])

    plot_label_crosstab(
        ('Edu score', ds['educational_score']),
        ('Toxicity score', ds['toxicity_score']),
        expe_path, "toxicity_edu")
    plot_label_crosstab(
        ('Toxicity score', ds['toxicity_score']),
        ('Topic', ds['topic']),
        expe_path, "topic_toxicity")
    plot_label_crosstab(
        ('Edu score', ds['educational_score']),
        ('Topic', ds['topic']),
        expe_path, "topic_edu")
    plot_histograms(ds, expe_path)

