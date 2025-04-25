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

datasets_by_language = {
    'en': [
        'dclm_dolmino',
        'open_web_math',
        'pes2o',
        'wikipedia_en',
    ],
    'fr': [
        'fineweb2_fra_Latn',
        'wikipedia_fr',
        'wikisource',
        'wiktionary', 
        'gallica_monographies',
        'gallica_press'
    ],
    'es': [
        'fineweb2_spa_Latn',
        'wikipedia_es'
    ],
    'de': [
        'fineweb2_deu_Latn',
        'wikipedia_de'
    ],
    'it': [
        'fineweb2_ita_Latn',
        'wikipedia_it'
    ],
    'code': [
        'algebraic_stack',
        'starcoder'
    ]
}

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
    # Define all possible weighting
    for k, v in datasets_by_language.items():
        parser.add_argument(
            f"--{k}",
            type=float,
            default=1.,
        )
        for d in v:
            parser.add_argument(
                f"--{d}",
                type=float,
                default=1.,
            )

    args = vars(parser.parse_args())
    print('Arguments:')
    pprint(args)
    hash = hash_dict(args)
    print(f'\nHash: {hash}')
    if args['name'] is None:
        name = hash
    else:
        name = args['name']
    data_path = args['data_path']

    # Read args and define each data weight
    upsampling = []
    for k, v in datasets_by_language.items():
        language_upsampling = args[k]
        for d in v:
            dataset_upsampling = args[d] * language_upsampling
            upsampling.append({"language": k, "dataset": d, "upsampling": dataset_upsampling})
    df_upsampling = pd.DataFrame(upsampling)
    
    # read data
    stats_df = pd.read_csv(os.path.join(data_path, "stats/all_stats_merged.csv"))
    df = df_upsampling.merge(stats_df, how='left', on='dataset')

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

    # Save the output to a JSON file
    output_dir = f"../datamix/{name}"
    os.makedirs(output_dir, exist_ok=True)
    # Save datamix
    with open(f"{output_dir}/datamix.json", 'w') as f:
        json.dump(out, f, indent=4)
    # Save Hash
    with open(f"{output_dir}/{hash}", "w", encoding="utf-8") as f:
        pass
    # Save datamix
    with open(f"{output_dir}/args.json", 'w') as f:
        json.dump(args, f, indent=4)
    # Save Language proportions
    language_df.to_csv(f"{output_dir}/language_proportion.csv")
    df.to_csv(f"{output_dir}/all_stats.csv")
