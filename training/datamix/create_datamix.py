import os
import argparse
import pandas as pd
import json
from pprint import pprint
import hashlib
from collections import OrderedDict
import re

def hash_dict(d):
    # Convert dict to a JSON string with sorted keys for consistency
    dict_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(dict_str.encode()).hexdigest()

dataset_info = pd.read_csv("datasets_info.csv")

def catch_name_and_cluster_size(name):
    pattern = r"(fineweb2_.*)_((cluster|edu)_.*)"
    match = re.search(pattern, name)
    if match:
        return match.group(1), match.group(2)
    else:
        return name, None
    
def apply_rehydratation(df, rehydratation_mapping):
    df[['dataset', 'group']] = df['name'].apply(lambda x: pd.Series(catch_name_and_cluster_size(x)))
    df['rehydratation_weight'] = df.apply(lambda x: rehydratation_mapping.get(x['group'], 1), axis=1)
    df['total_tokens_rehydrated'] = df['total_tokens'] * df['rehydratation_weight'] 

    df["group"] = pd.Categorical(
        df["group"], categories=list(rehydratation_mapping.keys()), ordered=True
    )
    df = df.sort_values('group')
    return df

if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(main_path, "data/tokens_ablation"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "ablations/datamix"),
        help="Output_dir",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        '--split_by',
        type=str,            
        default="cluster_size",
        choices = ["cluster_size", "edu_score"],
        help='Split by cluster size or by educational score'
    )
    parser.add_argument(
        '--rehydratation_weight',
        type=float,            
        nargs="+",
        default=None,
        help='Rehydratation_weights for cluster sizes or educational scores'
    )
    unique_category_and_language = set(dataset_info["language"]) | set(dataset_info["category"])
    if set(dataset_info["dataset"]) & unique_category_and_language:
        raise ValueError(f"Overlap between dataset and language/category!!!")
    for key in list(unique_category_and_language):
        parser.add_argument(
            f"--{key}",
            type=float,
            default=1.0,
            help="Category/language that you can upsample or downsample  (default value to 1)"
        )
    for key in dataset_info["dataset"]:
        parser.add_argument(
            f"--{key}",
            type=float,
            default=0.,
            help="Dataset weight (default value to 0!)"
        )

    args = parser.parse_args()
    data_path = args.data_path
    output_dir = args.output_dir
    name = args.name
    split_by = args.split_by
    rehydratation_weight = args.rehydratation_weight

    # Read args and define each data weight
    def compute_upsampling(row):
        if row["language"] != row["category"]:
            product = getattr(args, row["dataset"]) * getattr(args, row["language"]) * getattr(args, row["category"])
        else:
            product = getattr(args, row["dataset"]) * getattr(args, row["language"])
        return product

    dataset_info["upsampling"] = dataset_info.apply(compute_upsampling, axis=1)

    # Apply rehydratation if any
    stats_df = pd.read_csv(os.path.join(data_path, "stats/all_stats_merged.csv"))
    if split_by == "cluster_size":
        rehydratation_keys = ["cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5-100", "cluster_100-1000", "cluster_1000+"]
        rehydratation_weight = [1, 2, 3, 3, 5, 8, 1] if rehydratation_weight is None else rehydratation_weight
    elif split_by == "edu_score":
        rehydratation_keys = ["edu_0", "edu_1", "edu_2", "edu_3", "edu_4"]
        rehydratation_weight = [0, 1, 2, 3, 4] if rehydratation_weight is None else rehydratation_weight
    else:
        raise ValueError(f"Unknown split_by: {split_by}")
    assert len(rehydratation_keys) == len(rehydratation_weight)

    rehydratation_mapping = OrderedDict(
        zip(rehydratation_keys, rehydratation_weight)
    )
    stats_df = apply_rehydratation(stats_df, rehydratation_mapping)
    df = dataset_info.merge(stats_df, how="inner", on="dataset")

    # Upsampling
    df["total_tokens_upsampled"] = df["total_tokens_rehydrated"] * df["upsampling"]
    df["weight"] = df["total_tokens_upsampled"].transform(lambda x: x / x.sum())
    df = df[df["weight"] > 0]

    # Merge
    df["name"] = df["name"] + "_text_document"
    out = {
        "data_path": data_path,
        "train": df[["name", "weight"]].to_dict(orient="records"),
    }
    print("\nDatamix:")
    pprint(out)

    # Print language proportions
    print("\nLanguage proportion:")
    language_df = df.groupby("language")["weight"].sum()
    pprint(language_df)

    # Print Category proportions
    print("\nCategory proportion:")
    category_df = df.groupby("category")["weight"].sum()
    pprint(category_df)

    if name is not None:
        # Save the output to a JSON file
        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        # Save datamix
        with open(f"{output_dir}/datamix_{name}.json", "w") as f:
            json.dump(out, f, indent=4)
        # Save datamix
        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        # Save Language proportions
        language_df.to_csv(f"{output_dir}/language_proportion.csv")
        category_df.to_csv(f"{output_dir}/category_proportion.csv")
        df.to_csv(f"{output_dir}/all_stats.csv")
    else:
        print("\nYou should use --name if you want to save your datamix.")
