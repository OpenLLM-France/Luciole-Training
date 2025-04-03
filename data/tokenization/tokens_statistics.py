import json
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import glob
import re
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

def extract_token_lengths(data_path, name):
    suffix = "_text_document"
    dataset = indexed_dataset.MMapIndexedDataset(os.path.join(data_path, name + suffix))

    token_lengths = []
    for data in dataset:
        n = len(data)
        token_lengths.append(n)
    return token_lengths

def stats_summary(token_lengths, data_path, name):
    json.dump(
        {
            "total_tokens": np.sum(token_lengths).item(),
            "total_sequences": len(token_lengths),
            "min_tokens": min(token_lengths),
            "max_tokens": max(token_lengths),
            "mean_tokens": np.mean(token_lengths),
            "Q1_tokens": np.percentile(token_lengths, 25),  # First quartile (25th percentile)
            "Q2_tokens": np.median(token_lengths),
            "Q3_tokens": np.percentile(token_lengths, 75),  # Third quartile (75th percentile)
        },
        open(os.path.join(data_path, 'stats', f"{name}.json"), "w"),
        indent=4,
    )

def pdf_distribution(token_lengths, name, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()  # If no axes passed, use the current axes
    # Plot the KDE with log scale on the x-axis
    sns.kdeplot(token_lengths, bw_adjust=0.5, log_scale=(True, False), label=name, ax=ax, **kwargs)
    
    # Customize axis labels and title
    ax.set_xlabel("Token Lengths (log scale)")
    ax.set_ylabel("pdf")
    ax.set_xlim(10, 10**7)  # Set x-axis limit for log scale
    ax.set_xscale("log")

def process_and_save_plot(data_path, name, token_lengths, pattern=None):
    # Determine the output path based on whether it's a match or not
    output_path = os.path.join(data_path, 'figs', (pattern.replace(".*", "") + ".png") if pattern else f'{name}.png')
    
    if os.path.exists(output_path):
        print(f"Skip {('pattern ' + pattern if pattern else 'name ' + name)}.")
    else:
        # Create a new figure
        plt.figure(figsize=(8, 5))  
        # Set title depending on whether pattern is provided
        plt.title(pattern.replace(".*", "") if pattern else name)
        
        # Process the token lengths and generate the plot
        pdf_distribution(token_lengths, name, ax=plt.gca(), fill=bool(pattern), alpha=0.8 if pattern else 0.5)
        plt.legend()  # Add legend to distinguish between different plots
        

        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()  # Show the plot for non-matching names
        print(f"{('Pattern ' + pattern if pattern else 'Name ')}{name} processed.")

if __name__ == "__main__":
    data_path = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation"

    os.makedirs(os.path.join(data_path, 'stats'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'figs'), exist_ok=True)

    # Search for all matching files in the specified directory
    files = glob.glob(os.path.join(data_path, "*_text_document.idx"))

    # Extract the "name" part using regex
    names = [re.match(r"(.*?)_text_document\.idx", os.path.basename(f)).group(1) for f in files]

    # Run stats and fig
    patterns = [
        'fineweb2_fra.*',
        'fineweb2_ita.*',
        'fineweb2_deu.*',
        'fineweb2_spa.*',
        'wikipedia.*',
        'gallica.*'
    ]

    # Matched names processing
    matched_names_set = set()
    for pattern in patterns:
        compiled_pattern = re.compile(pattern)
        for name in names:
            if compiled_pattern.match(name) and name not in matched_names_set:
                print(f"Processing {name} for pattern {pattern}...")
                token_lengths = extract_token_lengths(data_path, name)
                stats_summary(token_lengths, data_path, name)
                matched_names_set.add(name)
                process_and_save_plot(data_path, name, token_lengths, pattern)

    # Non-matched names processing
    non_matching_names = [name for name in names if name not in matched_names_set]
    if non_matching_names:
        for name in non_matching_names:
            print(f"\nProcessing {name}...\n")
            token_lengths = extract_token_lengths(data_path, name)
            stats_summary(token_lengths, data_path, name)
            process_and_save_plot(data_path, name, token_lengths)