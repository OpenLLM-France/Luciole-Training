import os
import argparse
import pandas as pd
import json
from pprint import pprint
import re
from collections import OrderedDict
import sys


def apply_rehydratation(df, rehydratation_mapping):
    df["rehydratation_weight"] = df.apply(
        lambda x: rehydratation_mapping.get(x["group_name"], 1)
        if x["group_type"] == "cluster"
        else 1,
        axis=1,
    )
    df["total_tokens_rehydrated"] = df["total_tokens"] * df["rehydratation_weight"]
    df = df.sort_values(["name", "group_name"])
    return df


def to_nb_tokens(x):
    x = x.replace("b", " * 1_000_000_000")
    x = x.replace("m", " * 1_000_000")
    try:
        return int(eval(x))
    except Exception as e:
        raise ValueError(f"Invalid value for --mode: {x} (a number of tokens)") from e


def catch_name_and_cluster_size(name):
    pattern = "(fineweb2_.*_(cluster|edu))_(.*)"
    match = re.search(pattern, name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return name, None, None


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")

    first_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    first_parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(main_path, "data/tokenized_data/tokens_ablation_v1"),
        help="Path to the data directory",
    )
    first_parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "ablations/datamix"),
        help="Output directory",
    )
    first_parser.add_argument(
        "--name", type=str, default=None, help="Name of the output file"
    )
    first_parser.add_argument(
        "--target_tokens",
        type=str,
        default="35b",
        help="Target tokens (e.g., model size)",
    )
    first_parser.add_argument(
        "--rehydratation_weight",
        type=float,
        nargs="+",
        default=None,
        help="Rehydratation weights",
    )
    first_parser.add_argument("--help", "-h", action="store_true")

    # First pass to get the data path
    early_args, remaining_args = first_parser.parse_known_args()

    # Load your data (only if not just --help)
    df = pd.read_csv(os.path.join(early_args.data_path, "stats/all_stats_merged.csv"))

    def preprocess_entries(row):
        row["name"], row["group_type"], row["group_name"] = catch_name_and_cluster_size(
            row["name"]
        )
        return row

    df = df.apply(preprocess_entries, axis=1)

    # Add dataset-specific arguments
    for dataset_name in df["name"].unique():
        first_parser.add_argument(f"--{dataset_name}", type=float, default=0.0)

    # Show help *after* all arguments have been added
    if early_args.help:
        first_parser.print_help()
        sys.exit(0)

    # Final parse with the full set of arguments
    final_args = first_parser.parse_args()
    data_path = final_args.data_path
    rehydratation_weight = final_args.rehydratation_weight
    name = final_args.name
    output_dir = final_args.output_dir
    target_tokens = final_args.target_tokens

    # Apply rehydratation if any
    rehydratation_keys = [
        "1",
        "2",
        "3",
        "4",
        "5-100",
        "100-1000",
        "1000+",
    ]
    rehydratation_weight = (
        [1, 2, 3, 3, 5, 8, 1] if rehydratation_weight is None else rehydratation_weight
    )
    assert len(rehydratation_keys) == len(rehydratation_weight)

    rehydratation_mapping = OrderedDict(zip(rehydratation_keys, rehydratation_weight))

    print("Rehydratation mapping: ")
    pprint(rehydratation_mapping)

    df = apply_rehydratation(df, rehydratation_mapping)

    df["upsampling"] = df.apply(lambda row: getattr(final_args, row["name"]), axis=1)
    df["total_tokens_upsampled"] = df["total_tokens_rehydrated"] * df["upsampling"]
    df["weight"] = df["total_tokens_upsampled"].transform(lambda x: x / x.sum())
    df = df[df["weight"] > 0]

    # Calculating number of epochs per dataset
    df["epochs"] = df["weight"] * to_nb_tokens(target_tokens) / df["total_tokens"]
    print("\nData stats:")
    print(df)

    overtrained = df[df["epochs"] > 1]
    if not overtrained.empty:
        print(
            "\nWarning: The following datasets will be trained for more than one epoch:"
        )
        print(overtrained[["name", "total_tokens", "epochs"]])

    # Merge
    df["name"] = df["name"] + "_text_document"
    out = {
        "data_path": data_path,
        "train": df[["name", "weight"]].to_dict(orient="records"),
    }
    print("\nDatamix:")
    pprint(out)

    # Print language proportions
    # print("\nLanguage proportion:")
    # language_df = df.groupby("language")["weight"].sum()
    # pprint(language_df)

    # # Print Category proportions
    # print("\nCategory proportion:")
    # category_df = df.groupby("category")["weight"].sum()
    # pprint(category_df)

    if name is not None:
        # Save the output to a JSON file
        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        # Save datamix
        with open(f"{output_dir}/datamix_{name}.json", "w") as f:
            json.dump(out, f, indent=4)
        # Save datamix
        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(vars(final_args), f, indent=4)
        # Save Language proportions
        # language_df.to_csv(f"{output_dir}/language_proportion.csv")
        # category_df.to_csv(f"{output_dir}/category_proportion.csv")
        df.to_csv(f"{output_dir}/all_stats.csv")
    else:
        print("\nYou should use --name if you want to save your datamix.")
