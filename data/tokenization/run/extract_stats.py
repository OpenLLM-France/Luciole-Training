import json
import os
import numpy as np
import argparse
from nemo_patch import indexed_dataset


def extract_token_lengths(data_path, name):
    suffix = "_text_document"
    dataset = indexed_dataset.MMapIndexedDataset(os.path.join(data_path, name + suffix))
    return [len(data) for data in dataset]


def stats_summary(token_lengths):
    stats = {
        "total_tokens": np.sum(token_lengths).item(),
        "total_sequences": len(token_lengths),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "mean_tokens": np.mean(token_lengths),
        "Q1_tokens": np.percentile(token_lengths, 25),
        "Q2_tokens": np.median(token_lengths),
        "Q3_tokens": np.percentile(token_lengths, 75),
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="The path to the token directory (containing all the .idx and .bin)",
    )
    parser.add_argument(
        "name",
        type=str,
        help="The dataset name you want the statistics (end with _text_document.idx)",
    )
    args = parser.parse_args()
    data_path = args.data_path

    name = args.name.replace("_text_document.idx", "")

    stats_path = os.path.join(data_path, "stats", f"{name}.json")

    print(f"Extracting stats for {name}...")
    token_lengths = extract_token_lengths(data_path, name)
    os.makedirs(os.path.join(data_path, "stats"), exist_ok=True)
    stats = stats_summary(token_lengths)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Stats saved at {stats_path}.")
