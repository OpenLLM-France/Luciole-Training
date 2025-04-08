import json
import os
import numpy as np
import re
import glob
import pandas as pd
import argparse
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

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
        "--data_path", 
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation"
        )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-generation of stats even if they already exist."
        )
    args = parser.parse_args()
    data_path = args.data_path

    # Look for all files matching the pattern *_text_document.idx
    files = glob.glob(os.path.join(data_path, "*_text_document.idx"))
    names = [re.match(r"(.*?)_text_document\.idx", os.path.basename(f)).group(1) for f in files]

    for name in names:
        stats_path = os.path.join(data_path, 'stats', f"{name}.json")

        if not os.path.exists(stats_path) or args.force:
            print(f"Extracting and saving stats for {name}...")
            token_lengths = extract_token_lengths(data_path, name)
            os.makedirs(os.path.join(data_path, 'stats'), exist_ok=True)
            stats = stats_summary(token_lengths)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
        else:
            print(f"Stats for {name} already exist, skipping.")

    print("All stats saved.")
