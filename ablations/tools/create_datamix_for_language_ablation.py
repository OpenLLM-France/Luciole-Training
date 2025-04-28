import os
import argparse
import pandas as pd
import json

main_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data")

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
        "--input_dir",
        type=str,
        default="tokens_ablation",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Name of the output file",
    )
    parser.add_argument(
        "--no-rehydratation",
        action="store_true",
        help="If set, disables rehydratation."
    )
    args = parser.parse_args()

    language_weights = args.language_weights
    input_dir = args.input_dir
    suffix = args.suffix
    no_rehydratation = args.no_rehydratation
    if no_rehydratation:
        output_name = f"datamix_{'_'.join(language_weights)}_norehydratation"
    else:
        output_name = f"datamix_{'_'.join(language_weights)}"
    if suffix: 
        output_name += suffix
    output_name += ".json"
    
    assert len(language_weights) % 2 == 0
    
    if not os.path.exists(f"../datamix/{output_name}"):
        language_df = convert_args_to_dataframe(language_weights)

        # Load stats and normalize the total tokens by datasets
        stats_df = pd.read_csv(os.path.join(main_path, input_dir, "stats/all_stats_merged.csv"))
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
            'data_path': os.path.join(main_path, input_dir), 
            'train': df[['name', 'weight']].to_dict(orient='records'),
            'validation': [{'name': 'wikipedia_fr_text_document', 'weight': 0.5}, {'name': 'wikipedia_en_text_document', 'weight': 0.5}],
        }

        # Save the output to a JSON file
        with open(f"../datamix/{output_name}", 'w') as f:
            json.dump(out, f, indent=4)

    else: 
        print("Skipping datamix file creation, already exists")