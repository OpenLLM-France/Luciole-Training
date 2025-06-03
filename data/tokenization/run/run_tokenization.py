import os
import subprocess
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="datasets_to_tokenize.yaml",
        help=".yaml file that contains the datasets you want to tokenize. See for example datasets_to_tokenize.yaml.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(
            os.getenv("OpenLLM_OUTPUT"), "data/raw_data/data_for_ablation"
        ),
        help="Input directory that contains the processed datasets you want to tokenize. ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.getenv("OpenLLM_OUTPUT"), "data/tokenized_data/tokens_ablation"
        ),
        help="Output directory that will contain all your tokenized datasets, with name provided by your yaml file. You cannot use different tokenizer in one output_dir (it will raise an error).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="OpenLLM-France/Lucie-7B",
        help="The tokenizer you want to use to tokenize the data. This name will be saved in your output_dir.",
    )
    args = parser.parse_args()
    yaml_file = args.yaml_file
    raw_dataset_path = args.input_dir
    tokens_dataset_path = args.output_dir
    tokenizer_name = args.tokenizer_name

    os.makedirs(tokens_dataset_path, exist_ok=True)

    # Check if tokenizer name is already register, and if it match
    tokenizer_name_file = f"{tokens_dataset_path}/tokenizer_name.txt"
    if os.path.exists(tokenizer_name_file):
        with open(tokenizer_name_file, "r") as f:
            content = f.read()
            assert (
                tokenizer_name == content
            ), f"This output folder is associated with the tokenizer: {content}. You should either create a new output folder, or tokenize with the tokenizer {content}."
    else:
        with open(tokenizer_name_file, "w", encoding="utf-8") as f:
            f.write(tokenizer_name)

    # Load the YAML content
    with open(yaml_file, "r") as f:
        datasets = yaml.safe_load(f)["datasets"]

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
                subprocess.run(
                    [
                        "sbatch",
                        f"--job-name=tok_{name}",
                        "tokenize_one_dataset.slurm",
                        raw_path,
                        os.path.join(tokens_dataset_path, name),
                        tokenizer_name,
                    ]
                )
            else:
                print("--------------------------------------")
                print(f"⏩ Skipping {name}, already processed.")
                print("--------------------------------------")
        else:
            print("--------------------------------------")
            print(f"❌ Raw dataset not found for {name} at {raw_path}.")
            print("--------------------------------------")
