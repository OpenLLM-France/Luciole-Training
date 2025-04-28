import os
import subprocess
import yaml
import argparse

MAIN_PATH = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_file", type=str, default="datasets_to_tokenize.yaml"
    )
    parser.add_argument(
        "--input_dir", type=str, default="raw_datasets_ablation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="tokens_ablation"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="OpenLLM-France/Lucie-7B"
    )
    args = parser.parse_args()
    yaml_file = args.yaml_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    tokenizer_name = args.tokenizer_name

    # Define the path
    raw_dataset_path = os.path.join(MAIN_PATH, input_dir)
    tokens_dataset_path = os.path.join(MAIN_PATH, output_dir)
    os.makedirs(tokens_dataset_path, exist_ok=True)

    # Load the YAML content
    with open(yaml_file, "r") as f:
        datasets = yaml.safe_load(f)['datasets']

    # Iterate through each dataset entry
    for dataset in datasets:
        name = dataset.get("name")
        datapath = dataset.get("path")

        if not name or not datapath:
            continue  # Skip if required fields are missing

        raw_path = os.path.join(raw_dataset_path, datapath)
        output_idx = os.path.join(tokens_dataset_path, f"{name}_text_document.idx")

        if os.path.isdir(raw_path):
            if not os.path.isfile(output_idx):
                print("--------------------------------------")
                print(f"🚀 Processing dataset: {name}")
                print(f"📂 Path: {raw_path}")
                print("--------------------------------------")

                # Submit job using sbatch
                subprocess.run([
                    "sbatch",
                    f"--job-name=tok_{name}",
                    "tokenize_one_dataset.slurm",
                    raw_path,
                    os.path.join(tokens_dataset_path, name),
                    tokenizer_name
                ])
            else:
                print("--------------------------------------")
                print(f"⏩ Skipping {name}, already processed.")
                print("--------------------------------------")
        else:
            print("--------------------------------------")
            print(f"❌ Raw dataset not found for {name} at {raw_path}.")
            print("--------------------------------------")
