import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from data.tools.generate_token_stats import extract_token_lengths
import argparse

def pdf_distribution(token_lengths, name, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(token_lengths, bw_adjust=0.5, log_scale=(True, False), label=name, ax=ax, **kwargs)
    ax.set_xlabel("Token Lengths (log scale)")
    ax.set_ylabel("pdf")
    ax.set_xlim(10, 10**7)
    ax.set_xscale("log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-generation of plots even if they already exist.")
    args = parser.parse_args()

    data_path = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation"
    os.makedirs(os.path.join(data_path, 'figs'), exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_path, "*_text_document.idx")))
    names = [re.match(r"(.*?)_text_document\.idx", os.path.basename(f)).group(1) for f in files]

    patterns = [
        'fineweb2_fra.*',
        'fineweb2_ita.*',
        'fineweb2_deu.*',
        'fineweb2_spa.*',
        'wikipedia.*',
        'gallica.*'
    ]

    matched_names_set = set()
    for pattern in patterns:
        compiled_pattern = re.compile(pattern)
        matched_names = [name for name in names if compiled_pattern.match(name)]
        matched_names_set.update(matched_names)
    
        output_path = os.path.join(data_path, 'figs', pattern.replace(".*", "") + ".png")
        if os.path.exists(output_path) and not args.force:
            print(f"Plot already exists for pattern: {pattern}, skipping.")
            continue
        
        plt.figure(figsize=(8, 5))
        plt.title(pattern.replace(".*", ""))
        for name in matched_names:
            print(f"Plotting {name}...")
            token_lengths = extract_token_lengths(data_path, name)
            pdf_distribution(token_lengths, name, ax=plt.gca(), fill=False, alpha=0.8)
            plt.legend()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot for pattern: {pattern}")

    # Non-matching
    non_matching_names = [name for name in names if name not in matched_names_set]
    for name in non_matching_names:
        output_path = os.path.join(data_path, 'figs', f'{name}.png')
        if os.path.exists(output_path) and not args.force:
            print(f"Plot already exists for: {name}, skipping.")
            continue

        print(f"\nPlotting {name} (non-matching)...\n")
        token_lengths = extract_token_lengths(data_path, name)
        plt.figure(figsize=(8, 5))
        plt.title(name)
        pdf_distribution(token_lengths, name, ax=plt.gca(), fill=True, alpha=0.5)
        plt.legend()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved individual plot for: {name}")
