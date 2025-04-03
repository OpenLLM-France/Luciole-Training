import nemo_run as run
import fiddle as fdl
import os
import argparse

from nemo.collections import llm
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

# mamba
# pip install --user --no-cache-dir mamba-ssm[causal-conv1d]

# llama3_8b
# pip install --user --no-cache-dir zarr

# download tokenizer
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

TOKENIZER_NAME = "OpenLLM-France/Lucie-7B"


def configure_dataset(paths, seq_length=2048):
    tokenizer = get_tokenizer(tokenizer_name=TOKENIZER_NAME, use_fast=True)
    data = PreTrainingDataModule(
        paths=paths,
        global_batch_size=4,
        micro_batch_size=2,
        num_workers=8,
        pin_memory=True,
        seq_length=seq_length,
        tokenizer=tokenizer,
    )
    return data


def configure_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.llama3_8b.pretrain_recipe(
        name="iterating_dataloader",
        dir="iterating_dataloader",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.model.config.num_layers = 2
    recipe.trainer.max_steps = 5
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    executor = run.LocalExecutor(
        ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars
    )
    return executor


def run_dataloader(paths, output=None, number_of_data=1, seq_length=2048):
    recipe = configure_recipe(nodes=1, gpus_per_node=1)
    recipe.data = configure_dataset(paths, seq_length)
    print(f"recipe.trainer.devices={recipe.trainer.devices}")
    if output:
        os.makedirs(output, exist_ok=True)
    recipe.data.build(5, 1, 1, 1)
    recipe.data.trainer = fdl.build(recipe.trainer)
    for i, d in enumerate(recipe.data.train_dataloader()):
        print()
        print()
        print(f" START TEXT OF DATA {i}: ".center(80, "-"))
        ids = d["tokens"][0]
        text = recipe.data.tokenizer.ids_to_text(ids, remove_special_tokens=False)
        print(text)
        print(f" END TEXT OF DATA {i} ".center(80, "-"))
        if output:
            with open(os.path.join(output, f"{i}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
        if i + 1 >= number_of_data:
            break


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--paths",
        help="",
        default=["0.5", "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation/wikipedia_fr_text_document",
                 "0.5", "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation/wikipedia_es_text_document"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--output_path", help="out/", default=None, type=str)
    parser.add_argument(
        "--number_of_data", help="Number of iteration", default=10, type=str
    )
    parser.add_argument("--seq_length", help="", default=4096, type=str)
    args = parser.parse_args()
    run_dataloader(
        args.paths, args.output_path, args.number_of_data, args.seq_length
    )
