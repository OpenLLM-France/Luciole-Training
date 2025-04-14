import argparse
import subprocess
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_slurm_script(
    job_name, nodes, mode, config, output_dir, email, gpus_per_node=4
):
    # Choix des paramètres en fonction du mode
    if mode == "debug":
        qos = "qos_gpu_h100-dev"
        time = "00:30:00"
    elif mode == "20b" or mode == "35b":
        qos = "qos_gpu_h100-t3"
        time = "20:00:00"
    else:
        raise ValueError(f"Unkown mode {mode}, should be debug or 20b or 35b.")

    email_line = ""
    if email:
        email_line = f"""#SBATCH --mail-user={email}  # Où envoyer l'e-mail
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL            # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)"""

    # Contenu du script SLURM
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
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
echo "Nodes: {nodes}"
echo "Output dir: {output_dir}"

cwd=$(pwd)

export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output

export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface
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

echo "Arguments: {config} --num_nodes {nodes} --name {job_name} --mode {mode} --output_dir {output_dir}" 
srun torchrun $DISTRIBUTED_ARGS $cwd/train_llama.py {config} --num_nodes {nodes} --name {job_name} --mode {mode} --output_dir {output_dir}
"""
    return script


def submit_job(config, name_prefix, nodes, mode, output_dir, email):
    config = os.path.join("../datamix", config)
    if not os.path.exists(config):
        raise RuntimeError(f"Config : {config} does not exist")

    job_name = f"{os.path.splitext(os.path.basename(config))[0]}_{nodes}n_{mode}"
    if args.name_prefix:
        job_name = f"{name_prefix}_{job_name}"

    xp_output_dir = os.path.join(output_dir, job_name)

    slurm_script = create_slurm_script(
        job_name, nodes, mode, config, xp_output_dir, email
    )

    logger.info(f"Experiment name : {job_name}")
    logger.info(f"Experiment path : {xp_output_dir}")

    # Écrire le script dans un fichier temporaire
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
        print(f"Job submission failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mock.json")
    parser.add_argument("--name_prefix", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--mode", choices=["debug", "20b", "35b"], default="debug")
    parser.add_argument("--email", default=None)
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "ablations", "train"),
    )
    args = parser.parse_args()
    submit_job(
        args.config,
        args.name_prefix,
        args.num_nodes,
        args.mode,
        os.path.join(args.output_path, args.output_dir),
        args.email,
    )
