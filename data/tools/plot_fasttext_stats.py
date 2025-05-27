import os
import pandas as pd
import json
import matplotlib.pyplot as plt

main_path = os.getenv("OpenLLM_OUTPUT")
input_dir = os.path.join(
    main_path,
    "data/raw_datasets_ablation/fineweb2_filtered/fra_Latn/annotated_output/stats/fasttext_stats_1.0/output/",
)
output_dir = os.path.join(
    main_path,
    "data/raw_datasets_ablation/fineweb2_filtered/fra_Latn/annotated_output/stats/fasttext_stats_1.0/figs/",
)

## top by domain or topic
for group in ["fqdn", "topic"]:
    for fasttext in ["toxic", "edu"]:
        json_path = os.path.join(
            input_dir, group, f"fasttext_{fasttext}", "metric.json"
        )

        # Load JSON as dict of dicts
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.dropna().head(1000)
        df = df.sort_values(by="mean", ascending=False)
        top_df = df.head(50)

        # Create horizontal bar plot
        plt.figure(figsize=(10, 12))
        plt.barh(top_df.index, top_df["mean"], color="skyblue")
        plt.xlabel("Mean Score")
        plt.title(f"Top 50 Mean Scores for {fasttext}")
        plt.gca().invert_yaxis()  # Highest values at the top

        # Save figure
        save_path = os.path.join(output_dir, f"barh_{fasttext}_{group}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved barh plot to {save_path}")

## Cross tabulation
group = "edu_x_cluster"
fasttext = "toxic"
json_path = os.path.join(input_dir, group, f"fasttext_{fasttext}", "metric.json")

# Load JSON as dict of dicts
with open(json_path, "r") as f:
    data = json.load(f)
df = pd.DataFrame.from_dict(data, orient="index")

df.index.name = "index"
df = df.reset_index()
df[["edu_score", "cluster_size"]] = df["index"].str.extract(
    r"edu_score_(\d+)_cluster_size-(.*)"
)
print(df)

## histograms
group = "histogram"
for fasttext in ["toxic", "edu"]:
    json_path = os.path.join(input_dir, group, f"fasttext_{fasttext}", "metric.json")

    # Load JSON as dict of dicts
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = df.index.astype(float)

    # Values and their associated weights
    values = df.index.to_numpy()
    weights = df["total"].to_numpy()

    # Create histogram
    plt.hist(values, bins=50, weights=weights, edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram - {fasttext} score")
    plt.show()

    # Save figure
    save_path = os.path.join(output_dir, f"{group}_{fasttext}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved barh plot to {save_path}")
