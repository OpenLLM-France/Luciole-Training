import os
import subprocess
import logging
from pathlib import Path
from slurm_launcher import create_parser, pre_submit, write_launch_slurm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_slurm_conversion_script(job_id, xp_output_dir):
    job_name = f"conversion_of_{os.path.basename(xp_output_dir)}"
    set_env_path = Path(__file__).resolve().parent.parent
    set_env_path = f"{set_env_path}/set_env.sh"
    experiment_dir = xp_output_dir
    convert_script_path = Path(__file__).resolve().parent.parent
    convert_script_path = f"{convert_script_path}/conversion"
    dependency = f"#SBATCH --dependency=afterok:{job_id}" if job_id else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={xp_output_dir}/conversion/log_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{dependency}

source {set_env_path}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

DISTRIBUTED_ARGS=" \
        --nproc_per_node 1 \
        --nnodes $SLURM_NNODES \
        --node_rank $SLURM_PROCID \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
       "
echo $1
srun torchrun $DISTRIBUTED_ARGS {convert_script_path}/convert_experiment.py {experiment_dir}
"""
    return script


def submit_conversion(job_id, xp_output_dir):
    os.makedirs(os.path.join(xp_output_dir, "conversion"), exist_ok=True)
    if os.path.exists(os.path.join(xp_output_dir, "conversion", "completed.txt")):
        logger.info(
            f"Checkpoints already converted in {xp_output_dir}, skipping job submission. If you want to force submission, remove 'conversion/completed.txt'"
        )
        return None
    slurm_script = create_slurm_conversion_script(job_id, xp_output_dir)
    sbatch_script_path = os.path.join(xp_output_dir, "conversion/conversion.slurm")
    job_id = write_launch_slurm(sbatch_script_path, slurm_script)
    return job_id


def create_slurm_eval_script(job_id, xp_output_dir, task):
    job_name = f"evaluation_of_{job_id}"
    experiment_dir = xp_output_dir
    path_to_evaluation = Path(__file__).resolve().parent.parent
    path_to_evaluation = f"{path_to_evaluation}/evaluation"
    task_path = os.path.join(path_to_evaluation, "tasks", task)
    multilingual = False if task == "en" else True
    dependency = f"#SBATCH --dependency=afterok:{job_id}" if job_id else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={xp_output_dir}/evaluation/slurm_logs/pipeline_log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu
{dependency}

python {path_to_evaluation}/evaluate_experiment.py {experiment_dir} {task_path} {"--multilingual" if multilingual else ""}
"""
    return script


def submit_evaluation(job_id, xp_output_dir, task="en.txt"):
    slurm_script = create_slurm_eval_script(job_id, xp_output_dir, task)
    sbatch_script_path = os.path.join(
        xp_output_dir, f"evaluation/evaluation_{os.path.splitext(task)[0]}.slurm"
    )
    os.makedirs(os.path.join(xp_output_dir, "evaluation"), exist_ok=True)
    job_id = write_launch_slurm(sbatch_script_path, slurm_script)
    return job_id


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    try:
        job_id, xp_output_dir = pre_submit(args)
        conversion_id = submit_conversion(job_id, xp_output_dir)
        submit_evaluation(conversion_id, xp_output_dir, task="en.txt")
        submit_evaluation(conversion_id, xp_output_dir, task="fr.txt")
    except Exception:
        for jid in [job_id, conversion_id]:
            try:
                subprocess.run(["scancel", str(jid)], check=True)
                logger.info(f"Cancelled job {jid}")
            except subprocess.CalledProcessError as cancel_err:
                logger.warning(f"Failed to cancel job {jid}: {cancel_err}")
        logger.error("Job were cancelled because an error was raised!")
        raise
