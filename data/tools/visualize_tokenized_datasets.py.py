import re
import pandas as pd
import os
import argparse
import yaml 


def load_metadata():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the full path to the YAML file relative to the script
    yaml_file_path = os.path.join(script_dir, "../tokenization/datasets_to_tokenize.yaml")

    # Load the YAML content
    with open(yaml_file_path, "r") as f:
        out = yaml.safe_load(f)
    return out

def hbar_plot(df, label_column, output_path):
    df = df.groupby(label_column).agg({'total_tokens_rehydrated': 'sum'}).reset_index()
    df = df[df['total_tokens_rehydrated'] > 0]
    df['total_tokens_rehydrated'] /= 1e9

    df = df.sort_values(by='total_tokens_rehydrated', ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(df[label_column], df['total_tokens_rehydrated'])

    # Add labels and customize the plot
    plt.xscale('log')
    plt.xlabel('Total Tokens (in Billions)')
    plt.ylabel(label_column)
    plt.title(f'Total Tokens by {label_column}')

    # Optionally, add the labels on the bars
    for index, value in enumerate(df['total_tokens_rehydrated']):
        plt.text(value, index, f'{value:.2f} B', va='center', ha='left', fontsize=8, color='black')

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default=f"{main_path}/data/tokens_ablation/stats/all_stats_merged.csv"
        )
    parser.add_argument(
        "--output_path", 
        default=f"{main_path}/data/tokens_ablation/figs"
        )
    parser.add_argument('--multiplier', type=float, default=20, help="Multiplier for total tokens")
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(data_path)
    df['total_tokens'] *= args.multiplier
    print(df.head(5))
    
    ### Rehydratation upsampling
    for dataset, group in df.groupby('dataset'):
        if "fineweb2" in dataset:
            before = group['total_tokens'].sum()/ 1e9
            after = group['total_tokens_rehydrated'].sum()/ 1e9
            plt.figure(figsize=(12, 6))
            group.plot.bar(x="cluster_size", y=['total_tokens', 'total_tokens_rehydrated'], rot=0)
            plt.yscale('log')
            plt.title(f"Tokens per cluster size in {dataset}\n Total tokens: from {before:.1f} B to {after:.1f} B (x{after/before:.1f})")
            plt.savefig(os.path.join(output_path, f"rehydration_{dataset}.png"))

    ### Barplots
    metadata = load_metadata()
    metadata = pd.DataFrame(metadata['datasets'])
    metadata = metadata[["name", "category", "language"]]
    print(metadata.head(5))

    df = df.merge(metadata, how="left", on='name')

    # By dataset
    hbar_plot(df, 'dataset', os.path.join(output_path, "hbar_dataset.png"))
    hbar_plot(df, 'category', os.path.join(output_path, "hbar_category.png"))
    hbar_plot(df, 'language', os.path.join(output_path, "hbar_language.png"))
