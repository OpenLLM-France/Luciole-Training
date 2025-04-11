import os
import torch
import logging

from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.llm.gpt.model.llama import Llama31Config8B, Llama32Config1B

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def create_data(
    data_path, tokenizer_name="OpenLLM-France/Lucie-7B", batch_size=512, seq_length=2048
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    data = PreTrainingDataModule(
        paths=data_path,
        global_batch_size=batch_size,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=seq_length,  # 8192 for llama 32 1b
        tokenizer=tokenizer,
        split="1,0,0",
    )
    return data


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
