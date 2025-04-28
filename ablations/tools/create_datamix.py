import os
import argparse
import pandas as pd
import json
from pprint import pprint
import hashlib
import json

def hash_dict(d):
    # Convert dict to a JSON string with sorted keys for consistency
    dict_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(dict_str.encode()).hexdigest()

main_path = os.getenv("OpenLLM_OUTPUT")
default_data_path = os.path.join(main_path, "data/tokens_ablation/")

dataset_info = pd.read_csv('datasets_info.csv')

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
    unique_values = set(dataset_info['language']) | set(dataset_info['dataset']) | set(dataset_info['category'])
    unique_values = list(unique_values)
    for key in unique_values:
        parser.add_argument(
            f"--{key}",
            type=float,
            default=1.,
        )

    args = vars(parser.parse_args())
    print('Arguments:')
    pprint(args)
    hash = hash_dict(args)
    print(f'\nHash: {hash}')
    data_path = args['data_path']
    name = args['name']

    # Read args and define each data weight 
    def compute_upsampling(row):
        unique_values = {row['language'], row['dataset'], row['category']}  # a set: unique only
        product = 1.0
        for value in unique_values:
            product *= args.get(value, 1.0)  # default to 1.0 if value not found
        return product

    dataset_info['upsampling'] = dataset_info.apply(compute_upsampling, axis=1)
    
    # read data
    stats_df = pd.read_csv(os.path.join(data_path, "stats/all_stats_merged.csv"))
    df = dataset_info.merge(stats_df, how='left', on='dataset')

    total_tokens_ref = 'total_tokens_rehydrated'
    df['total_tokens_upsampled'] = df[total_tokens_ref] * df['upsampling']
    df['weight'] = df['total_tokens_upsampled'].transform(lambda x: x / x.sum())
    df = df[df['weight'] > 0]

    # Merge
    df['name'] = df['name'] + "_text_document"
    out = {
        'data_path': data_path, 
        'train': df[['name', 'weight']].to_dict(orient='records'),
    }
    print("\nDatamix:")
    pprint(out)

    # Print language proportions
    print("\nLanguage proportion:")
    language_df = df.groupby("language")['weight'].sum()
    pprint(language_df)

    # Print Category proportions
    print("\nCategory proportion:")
    category_df = df.groupby("category")['weight'].sum()
    pprint(category_df)

    if name is not None:
        # Save the output to a JSON file
        output_dir = f"../datamix/{name}"
        os.makedirs(output_dir, exist_ok=True)
        # Save datamix
        with open(f"{output_dir}/datamix_{name}.json", 'w') as f:
            json.dump(out, f, indent=4)
        # Save Hash
        with open(f"{output_dir}/{hash}", "w", encoding="utf-8") as f:
            pass
        # Save datamix
        with open(f"{output_dir}/args.json", 'w') as f:
            json.dump(args, f, indent=4)
        # Save Language proportions
        language_df.to_csv(f"{output_dir}/language_proportion.csv")
        category_df.to_csv(f"{output_dir}/category_proportion.csv")
        df.to_csv(f"{output_dir}/all_stats.csv")
    else:
        print("\nYou should use --name if you want to save your datamix.")