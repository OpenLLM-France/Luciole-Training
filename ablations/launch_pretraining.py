import nemo_run as run
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

DATA_PATH = os.getenv("OpenLLM_DATA")
OUTPUT_PATH = os.getenv("OpenLLM_OUTPUT")

TOKENIZER_NAME = "OpenLLM-France/Lucie-7B"

def configure_dataset():
    tokenizer = run.Config(get_tokenizer, tokenizer_name=TOKENIZER_NAME, use_fast=True)
    data = run.Config(
        PreTrainingDataModule,
        paths=DATA_PATH,
        global_batch_size=512,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=2048,    # 8192 for llama 32 1b
        tokenizer=tokenizer
    )
    return data

def configure_recipe(name, nodes: int = 1, gpus_per_node: int = 2):
    # recipe = llm.mamba2_130m.pretrain_recipe(
    recipe = llm.llama3_8b.pretrain_recipe(
        name=name,
        dir=os.path.join(OUTPUT_PATH, "nemo_experiments"),
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.data = configure_dataset()
    recipe.model.tokenizer = recipe.data.tokenizer
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }
    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor

def run_pretraining(xp_name):
    recipe = configure_recipe(xp_name, nodes=1, gpus_per_node=1)
    
    recipe.trainer.max_steps = 25_000
    
    recipe.model.config.share_embeddings_and_output_weights = True
    recipe.model.config.hidden_size = 2048
    recipe.model.config.ffn_hidden_size = 8192
    recipe.model.config.num_attention_heads = 32
    recipe.model.config.num_layers = 14
    recipe.model.config.num_query_groups = 32
    recipe.optim.config.lr=0.0003

    # Ok for 1 node:
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name=recipe.log.name)

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', help="", default="test", type=str)
    args = parser.parse_args()
    run_pretraining(args.name)