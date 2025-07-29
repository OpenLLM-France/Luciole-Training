import os
import json
import pandas as pd
import argparse


def merge_stats(data_path):
    # Get only JSON filenames
    json_files = [
        file for file in os.listdir(data_path) if file.lower().endswith(".json")
    ]

    data_list = []
    for json_file in json_files:
        with open(os.path.join(data_path, json_file), "r") as f:
            stats = json.load(f)
            stats = {"name": os.path.splitext(json_file)[0], **stats}
            data_list.append(stats)

    df = pd.DataFrame(data_list).sort_values(by="name")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "token_dir",
        type=str,
        help="Directory that contains all your tokenized datasets (.idx)",
    )
    args = parser.parse_args()
    token_dir = args.token_dir

    merged_df = merge_stats(os.path.join(token_dir, "stats"))
    for output_csv_path in [
        os.path.join(token_dir, "stats/all_stats_merged.csv"),
        "chronicles/all_stats_merged.csv",
    ]:
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Merged stats saved to {output_csv_path}")
