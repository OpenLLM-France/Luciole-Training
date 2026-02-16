import os
import pandas as pd
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot token treemaps by language and dataset."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="chronicles/raw",
        help="Path to the output directory. It must contains the repeats.csv file if you want to create a datamix.",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="chronicles/all_stats_merged.csv",
        help="Path to the all_stats.",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_lucie2",
        help="Path to the token directory.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the output instead of saving it to a file.",
    )
    args = parser.parse_args()
    stats_file = args.stats_file
    output_dir = args.output_dir
    repeats_file = os.path.join(output_dir, "repeats.csv")
    token_dir = args.token_dir

    assert os.path.isfile(
        repeats_file
    ), f"repeats.csv file not found in {output_dir}. Please make sure it exists and is named repeats.csv."

    # Read stats
    df = pd.read_csv(stats_file)
    assert (
        not df["name"].duplicated().any()
    ), f"Duplicate names in all_stats_merged.csv. Duplicates: {df[df['name'].duplicated()]['name'].tolist()}"

    # Merge repeats
    repeats = pd.read_csv(repeats_file)
    df = df.merge(repeats, on="name", how="left", validate="one_to_one")

    # Scale tokens
    df["total_tokens"] *= df["repeat"]

    # Rename datasets
    df["name"] = df["name"] + "_text_document"

    # Compute per-dataset weights
    df["weight"] = df["total_tokens"] // args.seq_length

    # Remove empty datasets
    df = df[df["weight"] > 0].copy()

    # Compute global stats AFTER filtering
    num_samples_global = df["weight"].sum()
    max_steps = num_samples_global // args.batch_size
    total_tokens_global = args.seq_length * args.batch_size * max_steps

    output = {
        "data_path": token_dir,
        "total_tokens": int(total_tokens_global),
        "train": (df[["name", "weight"]].sort_values("name").to_dict(orient="records")),
    }

    if args.debug:
        print(output)
    else:
        with open(f"{output_dir}/datamix.json", "w") as f:
            json.dump(output, f, indent=4)
