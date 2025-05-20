import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def read_datamix_file(file):
    loaded_data = None
    if file.endswith(".json"):
        import json

        with open(file, "r") as f:
            loaded_data = json.load(f)
    elif file.endswith(".yaml"):
        import yaml

        with open(file, "r") as f:
            loaded_data = yaml.safe_load(f)
    else:
        raise RuntimeError(f"Config should be a json or a yaml, got {file}")

    def make_data_flattened_list(split="train"):
        data_paths = []
        for dataset in loaded_data.get(split, []):
            data_paths.append(str(dataset["weight"]))
            data_paths.append(os.path.join(loaded_data["data_path"], dataset["name"]))
        return data_paths

    logger.info(loaded_data)
    if "validation" in loaded_data:
        data_paths = {
            "train": make_data_flattened_list("train"),
            "validation": make_data_flattened_list("validation"),
            "test": make_data_flattened_list("validation"),
        }
    else:
        data_paths = make_data_flattened_list("train")
    # logger.info(">>>>>>>>>>>")
    # logger.info(data_paths)

    # Read tokenizer
    try:
        with open(
            os.path.join(loaded_data["data_path"], "tokenizer_name.txt"), "r"
        ) as f:
            tokenizer_name = f.read().strip()
            logger.info(f"Find tokenizer: {tokenizer_name}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"tokenizer_name.txt not found in {loaded_data['data_path']}. Please rerun the tokenization step."
        )
    return data_paths, tokenizer_name
