import os
import torch
import logging

from nemo.collections.llm.gpt.model.llama import Llama31Config8B, Llama32Config1B

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

    if "valid" in loaded_data:
        data_paths = {
            "train": make_data_flattened_list("train"),
            "validation": make_data_flattened_list("validation"),
            "test": make_data_flattened_list("test"),
        }
    else:
        data_paths = make_data_flattened_list("train")
    return data_paths


def get_config(size_1b=True):
    # return Llama32Config1B()
    if not size_1b:
        return Llama31Config8B()
    else:
        return Llama32Config1B()
