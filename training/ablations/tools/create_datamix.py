import os
import argparse
import pandas as pd
import json
from pprint import pprint
import hashlib
from collections import OrderedDict

def hash_dict(d):
    # Convert dict to a JSON string with sorted keys for consistency
    dict_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(dict_str.encode()).hexdigest()


main_path = os.getenv("OpenLLM_OUTPUT")
default_data_path = os.path.join(main_path, "data/tokens_ablation/")

dataset_info = pd.read_csv("datasets_info.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        '--rehydratation_weight',
        type=float,            
        nargs=7,
        default=[1, 2, 3, 3, 5, 8, 1],
        help='Rehydratation_weights for cluster sizes: ["1", "2", "3", "4", "5-100", "100-1000", "1000+"]'
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

    args = vars(parser.parse_args())
    print("Arguments:")
    pprint(args)
    # hash = hash_dict(args)
    # print(f"\nHash: {hash}")
    data_path = args["data_path"]
    name = args["name"]

    # Read args and define each data weight
    def compute_upsampling(row):
        if row["language"] != row["category"]:
            product = args.get(row["dataset"]) * args.get(row["language"]) * args.get(row["category"])
        else:
            product = args.get(row["dataset"]) * args.get(row["language"])
        return product

    dataset_info["upsampling"] = dataset_info.apply(compute_upsampling, axis=1)

    # read data
    stats_df = pd.read_csv(os.path.join(data_path, "stats/all_stats_merged.csv"))
    df = dataset_info.merge(stats_df, how="left", on="dataset")

    if True:
        rehydratation_mapping = OrderedDict(
            zip(["1", "2", "3", "4", "5-100", "100-1000", "1000+"], args["rehydratation_weight"])
        )

        df = df.drop(labels=['rehydratation_weight', 'total_tokens_rehydrated'], axis=1)
        df['rehydratation_weight'] = df.apply(lambda x: rehydratation_mapping.get(x['cluster_size'], 1), axis=1)
        df['total_tokens_rehydrated'] = df['total_tokens'] * df['rehydratation_weight'] 

    total_tokens_ref = "total_tokens_rehydrated"
    df["total_tokens_upsampled"] = df[total_tokens_ref] * df["upsampling"]
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
        output_dir = f"../datamix/{name}"
        os.makedirs(output_dir, exist_ok=True)
        # Save datamix
        with open(f"{output_dir}/datamix_{name}.json", "w") as f:
            json.dump(out, f, indent=4)
        # # Save Hash
        # with open(f"{output_dir}/{hash}", "w", encoding="utf-8") as f:
        #     pass
        # Save datamix
        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(args, f, indent=4)
        # Save Language proportions
        language_df.to_csv(f"{output_dir}/language_proportion.csv")
        category_df.to_csv(f"{output_dir}/category_proportion.csv")
        df.to_csv(f"{output_dir}/all_stats.csv")
    else:
        print("\nYou should use --name if you want to save your datamix.")
