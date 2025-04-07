import nemo_run as run
import os
import argparse

from typing import Optional
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


def configure_dataset(data_path):
    tokenizer = run.Config(get_tokenizer, tokenizer_name=TOKENIZER_NAME, use_fast=True)
    data = run.Config(
        PreTrainingDataModule,
        paths=data_path,
        global_batch_size=512,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=2048,  # 8192 for llama 32 1b
        tokenizer=tokenizer,
    )
    return data


def configure_recipe(
    name, data_path, output_path, nodes: int = 1, gpus_per_node: int = 2
):
    # recipe = llm.mamba2_130m.pretrain_recipe(
    recipe = llm.llama3_8b.pretrain_recipe(
        name=name,
        dir=os.path.join(output_path, "nemo_experiments"),
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.data = configure_dataset(data_path)
    recipe.model.tokenizer = recipe.data.tokenizer
    return recipe


def slurm_executor(
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    output_path: str,
    time: str = "00:10:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    # container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
    custom_srun_args: Optional[list[str]] = None,
) -> run.SlurmExecutor:
    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "0",
        "TOKENIZERS_PARALLELISM": "False",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "0",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "NEMORUN_HOME": output_path,  # should be job_dir minus experiments
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    srun_args = ["--mpi=pmix"]
    if custom_srun_args:
        srun_args.extend(custom_srun_args)

    tunnel = run.LocalTunnel(
        job_dir=os.path.join(output_path, "experiments")
    )  # /!\ must be experiments and not something else
    packager = run.GitArchivePackager(
        basepath="/linkhome/rech/gendjf01/uxn76rc/llm/OpenLLM-BPI-Training"
    )
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=tunnel,
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        # mem="0",
        qos="qos_gpu_a100-dev",
        exclusive=True,  # should be true
        constraint="a100",
        job_name_prefix="testing_singularity",
        gres="gpu:1",
        # job_dir=os.path.join(output_path, "nemo_experiments"),
        # launcher="torchrun",
        packager=packager,
    )

    executor.container_image = (
        "/lustre/fsn1/singularity/images/uxn76rc/image-singularity-nemo.sif"
    )
    executor.container_mounts = []
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def my_slurm_executor(output_path):
    # TODO: Set your custom parameters for the Slurm Executor.
    return slurm_executor(
        account="qgz@a100",
        partition="gpu_p5",
        nodes=1,
        devices=1,
        output_path=output_path,
    )


def run_pretraining(xp_name, data_path, output_path):
    recipe = configure_recipe(xp_name, data_path, output_path, nodes=1, gpus_per_node=1)

    recipe.trainer.max_steps = 3  # 25_000

    recipe.model.config.share_embeddings_and_output_weights = True
    recipe.model.config.hidden_size = 2048
    recipe.model.config.ffn_hidden_size = 8192
    recipe.model.config.num_attention_heads = 32
    recipe.model.config.num_layers = 14
    recipe.model.config.num_query_groups = 32
    recipe.optim.config.lr = 0.0003

    executor = my_slurm_executor(output_path)
    # Ok for 1 node:
    # executor = my_slurm_executor(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    # run.run(recipe, executor=my_slurm_executor(output_path), name=recipe.log.name)
    with run.Experiment(recipe.log.name) as exp:
        exp.add(
            recipe,
            executor=executor,
        )
        exp.run(tail_logs=True)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_path",
        help="Data",
        default="/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/Wikipedia--fr_text_document",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Data",
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output",
        type=str,
    )
    parser.add_argument("--name", help="", default="test", type=str)
    args = parser.parse_args()
    run_pretraining(args.name, args.data_path, args.output_path)
