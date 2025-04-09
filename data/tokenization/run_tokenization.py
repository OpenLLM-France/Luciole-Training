import os
import subprocess
import yaml

YAML_FILE = "datasets_to_tokenize.yaml"
MAIN_PATH = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data")

raw_dataset_path = os.path.join(MAIN_PATH, "raw_datasets_ablation")
tokens_dataset_path = os.path.join(MAIN_PATH, "tokens_ablation")

os.makedirs(tokens_dataset_path, exist_ok=True)

# Load the YAML content
with open(YAML_FILE, "r") as f:
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
                os.path.join(tokens_dataset_path, name)
            ])
        else:
            print("--------------------------------------")
            print(f"⏩ Skipping {name}, already processed.")
            print("--------------------------------------")
    else:
        print("--------------------------------------")
        print(f"❌ Raw dataset not found for {name} at {raw_path}.")
        print("--------------------------------------")
