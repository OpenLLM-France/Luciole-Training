import nemo_run as run
from nemo.collections import llm
import fiddle as fdl

from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

# mamba
# pip install --user --no-cache-dir mamba-ssm[causal-conv1d]

# llama3_8b
# pip install --user --no-cache-dir zarr

# download tokenizer
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
def configure_dataset():
    tokenizer = get_tokenizer(tokenizer_name="OpenLLM-France/Lucie-7B", use_fast=True)
    data =  PreTrainingDataModule(
        paths='/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/Wikipedia--fr_text_document',
        global_batch_size=4,
        micro_batch_size=2,
        num_workers=8,
        pin_memory=True,
        seq_length=2048,
        tokenizer=tokenizer
    )
    return data

def configure_recipe(nodes: int = 1, gpus_per_node: int = 2):
    # recipe = llm.mamba2_130m.pretrain_recipe(
    recipe = llm.llama3_8b.pretrain_recipe(
        name="test_llama",
        dir="/lustre/fsn1/projects/rech/qgz/uxn76rc/llm/xp",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.model.config.num_layers = 2
    recipe.trainer.max_steps = 5
    
    # recipe.model.tokenizer.model_name = "OpenLLM-France/Lucie-7B"
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

def run_dataloader(number_of_data=1):
    recipe = configure_recipe(nodes=1, gpus_per_node=2)
    recipe.data = configure_dataset()
    print(f"recipe.trainer.devices={recipe.trainer.devices}")

    recipe.data.build(5, 1, 1, 1)
    recipe.data.trainer=fdl.build(recipe.trainer)
    for i, d in enumerate(recipe.data.train_dataloader()):
        print()
        print()
        print(f" START TEXT OF DATA {i}: ".center(80, '-'))
        ids = d['tokens'][0]
        text = recipe.data.tokenizer.ids_to_text(ids, remove_special_tokens=False)
        print(text)
        print(f" END TEXT OF DATA {i} ".center(80, '-'))
        if i+1>=number_of_data:
            break

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_dataloader()