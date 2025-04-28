import os
import argparse
import pandas as pd
import json

main_path = os.getenv("OpenLLM_OUTPUT")
default_data_path = os.path.join(main_path, "data/tokens_ablation/")

def convert_args_to_dataframe(language_weights):
    out = []
    for i in range(0, len(language_weights), 2):
        language_weight = float(language_weights[i])
        language = language_weights[i + 1]

        if language in ["fra_Latn", "deu_Latn", "ita_Latn", "esp_Latn"]:
            out.append({
                'dataset': f"fineweb2_{language}",
                'language_weight': language_weight,
                })
        elif language == "eng_Latn":
            out.append({
                'dataset': "fineweb_edu",
                'language_weight': language_weight,
                })
        else:
            NotImplementedError(f"Language {language} not implemented")
    return pd.DataFrame(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language_weights", 
        default=[".5", "fra_Latn", ".5", "eng_Latn"],
        nargs="+",
        )
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--expe_name",
        type=str,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        "--no-rehydratation",
        action="store_true",
        help="If set, disables rehydratation."
    )
    args = parser.parse_args()

    language_weights = args.language_weights
    data_path = args.data_path
    expe_name = args.expe_name
    no_rehydratation = args.no_rehydratation
    if expe_name is None:
        if no_rehydratation:
            expe_name = f"datamix_{'_'.join(language_weights)}_norehydratation.json"
        else:
            expe_name = f"datamix_{'_'.join(language_weights)}.json"
    
    assert len(language_weights) % 2 == 0
    
    if not os.path.exists(f"../datamix/{expe_name}"):
        language_df = convert_args_to_dataframe(language_weights)

        # Load stats and normalize the total tokens by datasets
        stats_df = pd.read_csv(os.path.join(data_path, "stats/all_stats_merged.csv"))
        if no_rehydratation:
            total_tokens_ref = 'total_tokens'
        else:
            total_tokens_ref = 'total_tokens_rehydrated'
        stats_df['weight_per_dataset'] = stats_df.groupby('dataset')[total_tokens_ref].transform(lambda x: x / x.sum())

        # Merge
        df = pd.merge(stats_df, language_df, on='dataset', how='inner')
        df['weight'] = df['weight_per_dataset'] * df['language_weight']
        df['name'] = df['name'] + "_text_document"

        out = {
            'data_path': data_path, 
            'train': df[['name', 'weight']].to_dict(orient='records'),
            'validation': [{'name': 'wikipedia_fr_text_document', 'weight': 0.5}, {'name': 'wikipedia_en_text_document', 'weight': 0.5}],
        }

        # Save the output to a JSON file
        with open(f"../datamix/{expe_name}", 'w') as f:
            json.dump(out, f, indent=4)

    else: 
        print("Skipping datamix file creation, already exists")