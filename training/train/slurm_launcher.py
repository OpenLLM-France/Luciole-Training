import argparse
import subprocess
import os
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_slurm_script(
    job_name,
    email,
    output_dir,
    config,
    arch,
    num_nodes,
    gpus_per_node,
    mode,
    fp8,
    tensor_parallelism,
    pipeline_parallelism,
):
    # Choix des paramètres en fonction du mode
    if mode == "debug" or mode == "benchmark":
        qos = "qos_gpu_h100-dev"
        time = "00:15:00"
    elif mode == "20b" or mode == "35b":
        qos = "qos_gpu_h100-t3"
        time = "20:00:00"
    else:
        raise ValueError(f"Unkown mode {mode}, should be debug, benchmark, 20b or 35b.")

    email_line = ""
    if email:
        email_line = f"""#SBATCH --mail-user={email}  # Où envoyer l'e-mail
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL            # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)"""

    train_path = Path(__file__).resolve().parent

    logger.info(f"Train script path: {train_path}/train.py")

    if arch == "llama":  # backward compatibility
        arch = "llama1b"

    args = f"{config} --arch {arch} --num_nodes {num_nodes} --name {job_name} --mode {mode} --output_dir {output_dir} --num_gpus_per_node {gpus_per_node}"
    if fp8:
        args += " --fp8"
    if tensor_parallelism:
        args += f" --tensor_parallelism {tensor_parallelism}"
    if pipeline_parallelism:
        args += f" --pipeline_parallelism {pipeline_parallelism}"

    # Contenu du script SLURM
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time={time}
#SBATCH --output={output_dir}/log_%j.out 
#SBATCH --hint=nomultithread 
#SBATCH --qos={qos}
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{email_line}

echo "Job name: {job_name}"
echo "Qos: {qos}"
echo "Time limit: {time}"
echo "Mode: {mode}"
echo "Nodes: {num_nodes}"
echo "Output dir: {output_dir}"

cwd=$(pwd)

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export TOKENIZERS_PARALLELISM=false

module purge
module load arch/h100 nemo/2.1.0

# Set environment variables for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE={gpus_per_node}  # Adjust based on your setup

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank $SLURM_PROCID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       "

echo "Arguments: {args}" 
srun torchrun $DISTRIBUTED_ARGS {train_path}/train.py {args}
"""
    return script


def submit_job(**kwargs):
    config = kwargs["config"]
    if not os.path.exists(config):
        tested_config = os.path.join("../ablations/datamix", config)
        if not os.path.exists(tested_config):
            raise RuntimeError(f"Config: {config} does not exist")
        config = tested_config

    config_name = os.path.splitext(os.path.basename(config))[0]
    job_name_parts = [
        kwargs["arch"],
        config_name,
        f'{kwargs["num_nodes"]}n',
        kwargs["mode"],
    ]
    if kwargs.get("fp8"):
        job_name_parts.append("fp8")
    if kwargs.get("name_prefix"):
        job_name_parts.insert(0, kwargs["name_prefix"])
    if kwargs.get("tensor_parallelism"):
        job_name_parts.append(f"tp{kwargs['tensor_parallelism']}")
    if kwargs.get("pipeline_parallelism"):
        job_name_parts.append(f"pp{kwargs['pipeline_parallelism']}")
    job_name = "_".join(job_name_parts)

    xp_output_dir = os.path.join(kwargs["output_dir"], job_name)
    os.makedirs(xp_output_dir, exist_ok=True)

    args = {
        **kwargs,
        "job_name": job_name,
        "config": config,
        "output_dir": xp_output_dir,
    }
    args.pop("name_prefix")
    slurm_script = create_slurm_script(**args)

    logger.info(f"Experiment name : {job_name}")
    logger.info(f"Experiment path : {xp_output_dir}")

    sbatch_script_path = os.path.join(xp_output_dir, "launch.slurm")
    os.makedirs(xp_output_dir, exist_ok=True)
    with open(sbatch_script_path, "w") as fout:
        fout.write(slurm_script)
    logger.info(f"Generated slurm script : {sbatch_script_path}")

    shutil.copy2(config, xp_output_dir)
    logger.info(f"Copied datamix file : {config} to {xp_output_dir}")

    try:
        subprocess.run(["sbatch", sbatch_script_path], check=True)
        logger.info("Job submitted")
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mock.json")
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=["llama", "llama1b", "llama8b", "mamba"],
    )
    parser.add_argument("--name_prefix", default="", type=str)
    parser.add_argument("--email", default=None)
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "ablations", "train"),
    )
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus_per_node", default=4, type=int)
    parser.add_argument("--mode", default="debug", type=str)
    parser.add_argument("--fp8", default=False, action="store_true")
    parser.add_argument("--tensor_parallelism", default=None, type=int)
    parser.add_argument("--pipeline_parallelism", default=None, type=int)
    args = parser.parse_args()

    args_dict = vars(args)
    args_dict["output_dir"] = os.path.join(args.output_path, args.output_dir)
    args_dict.pop("output_path")

    submit_job(**args_dict)
