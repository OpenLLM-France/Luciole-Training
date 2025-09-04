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


def extract_info(text):
    splitted_text = text.split("_")
    assert len(splitted_text) <= 3, f"Error in name format, too much _ in {text}"
    dataset = splitted_text[0]
    language = splitted_text[-1]
    if len(splitted_text) == 3:
        subset = splitted_text[1]
    else:
        subset = None
    return {"dataset": dataset, "subset": subset, "language": language}


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
    merged_df = pd.concat(
        [merged_df, merged_df["name"].apply(extract_info).apply(pd.Series)], axis=1
    )
    merged_df = merged_df.sort_values(["language", "dataset"], ascending=True)

    for output_csv_path in [
        os.path.join(token_dir, "stats/all_stats_merged.csv"),
        "chronicles/all_stats_merged.csv",
    ]:
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Merged stats saved to {output_csv_path}")
