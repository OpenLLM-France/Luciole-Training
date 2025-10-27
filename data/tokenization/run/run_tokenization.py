import os
import subprocess
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file that contains the datasets you want to tokenize. See for example configs/ablations_v0.yaml.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The tokenizer you want to use to tokenize the data. This name will be saved in the output_dir.",
    )
    parser.add_argument(
        "--start_with",
        type=str,
        default="",
        help="Tokenize only datasets whose name start with this value",
    )
    parser.add_argument(
        "--remove-prefix",
        action="store_true",
        help="Remove prefix to the data.",
    )
    args = parser.parse_args()
    yaml_file = args.yaml_file
    tokenizer_name = args.tokenizer_name
    start_with = args.start_with

    # Load the YAML config
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Read output dir
    tokens_dataset_path = yaml_data["output_dir"]
    os.makedirs(tokens_dataset_path, exist_ok=True)

    # Check if tokenizer name is already register, and if it match
    tokenizer_name_file = f"{tokens_dataset_path}/tokenizer_name.txt"
    if os.path.exists(tokenizer_name_file):
        with open(tokenizer_name_file, "r") as f:
            content = f.read()
            if tokenizer_name is None:
                tokenizer_name = content
                print(
                    f"Warning: No tokenizer name provided, using the one from {tokenizer_name_file}: {content}"
                )
            else:
                assert (
                    tokenizer_name == content
                ), f"This output folder is associated with the tokenizer: {content}. You should either create a new output folder, or tokenize with the tokenizer {content}."
    else:
        assert (
            tokenizer_name is not None
        ), "You must provide a tokenizer name if it is not already registered."
        with open(tokenizer_name_file, "w", encoding="utf-8") as f:
            f.write(tokenizer_name)

    # Iterate through each dataset entry
    for dataset in yaml_data["datasets"]:
        name = dataset["name"]
        raw_path = dataset["path"]
        regex_filter = dataset.get("regex", r".*\.json.*")

        output_idx = os.path.join(tokens_dataset_path, f"{name}_text_document.idx")
        output_bin = os.path.join(tokens_dataset_path, f"{name}_text_document.bin")

        if os.path.isdir(raw_path):
            if not os.path.isfile(output_idx) and name.startswith(args.start_with):
                if os.path.isfile(output_bin):
                    print("--------------------------------------")
                    print(
                        f"⚠️  Warning for {name}! Found a .bin file at {output_bin}, but no .idx file. Either a job has failed or is still running."
                    )
                    print("--------------------------------------")
                else:
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
                            regex_filter,
                            "--remove-prefix" if args.remove_prefix else "",
                        ]
                    )
            else:
                print("--------------------------------------")
                print(f"⏩ Skipping {name}")
                print("--------------------------------------")
        else:
            print("--------------------------------------")
            print(f"❌ Raw dataset not found for {name} at {raw_path}.")
            print("--------------------------------------")
