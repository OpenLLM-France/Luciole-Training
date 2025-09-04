import os
import subprocess
import logging
import argparse
from pathlib import Path
from slurm_launcher import (
    create_parser,
    pre_submit,
    write_launch_slurm,
    generate_email_line,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_slurm_conversion_script(job_id, xp_output_dir, email=None):
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
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{dependency}
{generate_email_line(email)}

source {set_env_path}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + $SLURM_JOB_ID))

DISTRIBUTED_ARGS=" \
        --nproc_per_node 1 \
        --nnodes $SLURM_NNODES \
        --node_rank $SLURM_PROCID \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --rdzv_id $SLURM_JOB_ID \
        --max_restarts 0 \
       "
echo $1
srun torchrun $DISTRIBUTED_ARGS {convert_script_path}/convert_experiment.py {experiment_dir}
"""
    return script


def submit_conversion(job_id, xp_output_dir, email=None):
    os.makedirs(os.path.join(xp_output_dir, "conversion"), exist_ok=True)
    if os.path.exists(os.path.join(xp_output_dir, "conversion", "completed.txt")):
        logger.info(
            f"Checkpoints already converted in {xp_output_dir}, skipping job submission. If you want to force submission, remove 'conversion/completed.txt'"
        )
        return None
    slurm_script = create_slurm_conversion_script(job_id, xp_output_dir, email=email)
    sbatch_script_path = os.path.join(xp_output_dir, "conversion/conversion.slurm")
    job_id = write_launch_slurm(sbatch_script_path, slurm_script, task="conversion")
    return job_id


def create_slurm_eval_script(
    job_id,
    xp_output_dir,
    task,
    email=None,
    command="accelerate",
    fineweb=False,
    skip_existing_evals=True,
):
    job_name = f"evaluation_of_{job_id}_{os.path.basename(xp_output_dir)}"
    experiment_dir = xp_output_dir
    path_to_evaluation = Path(__file__).resolve().parent.parent
    path_to_evaluation = f"{path_to_evaluation}/evaluation"
    task_path = os.path.join(path_to_evaluation, "tasks", task)
    multilingual = True if task not in ["en.txt", "fineweb2.txt"] else False
    fineweb = True if task == "fineweb2.txt" else False
    dependency = f"#SBATCH --dependency=afterok:{job_id}" if job_id else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={xp_output_dir}/evaluation/slurm_logs/pipeline_log_{task.replace(".txt", "")}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --account=qgz@cpu
{dependency}
{generate_email_line(email)}

python {path_to_evaluation}/evaluate_experiment.py {experiment_dir} {task_path} --command {command} {"--skip_existing" if skip_existing_evals else ""} {"--multilingual" if multilingual else ""} {"--fineweb" if fineweb else ""} 
"""
    return script


def submit_evaluation(
    job_id,
    xp_output_dir,
    task="en.txt",
    email=None,
    command="accelerate",
    skip_existing_evals=True,
):
    os.makedirs(os.path.join(xp_output_dir, "evaluation"), exist_ok=True)
    if os.path.exists(os.path.join(xp_output_dir, "evaluation", "completed.txt")):
        logger.info(
            f"Evaluations already done for {xp_output_dir}, skipping job submission. If you want to force submission, remove 'conversion/completed.txt'"
        )
        return None
    slurm_script = create_slurm_eval_script(
        job_id,
        xp_output_dir,
        task,
        email=email,
        command=command,
        skip_existing_evals=skip_existing_evals,
    )
    sbatch_script_path = os.path.join(
        xp_output_dir, f"evaluation/evaluation_{os.path.splitext(task)[0]}.slurm"
    )
    job_id = write_launch_slurm(sbatch_script_path, slurm_script, task="evaluation")
    return job_id


def update_email_args(step, source_args):
    args = argparse.Namespace(**vars(source_args))
    if step == "train":
        if args.email_when == "all" or args.email_when == "train":
            args.email = args.email
        else:
            args.email = None
        del args.email_when
        return args
    if step == "eval" and (args.email_when == "all" or args.email_when == "eval"):
        return args.email
    if step == "conversion" and (
        args.email_when == "all" or args.email_when == "conversion"
    ):
        return args.email
    else:
        return None


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--email_when",
        type=str,
        choices=["train", "conversion", "eval", "all"],
        help="At which step do you want to send emails",
        default="all",
    )
    parser.add_argument(
        "--tasks",
        help="Tasks to evaulate on, skip if empty. If multiple, separate by space",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--skip_existing_evals",
        action="store_true",
        help="Skip evaluations if evaluation results already exist (even if tasks are different).",
        default=False,
    )
    source_args = parser.parse_args()
    tasks = source_args.tasks
    skip_existing_evals = source_args.skip_existing_evals
    job_id = None
    conversion_id = None
    del source_args.tasks
    del source_args.skip_existing_evals
    try:
        job_id, xp_output_dir = pre_submit(update_email_args("train", source_args))
        conversion_id = submit_conversion(
            job_id, xp_output_dir, email=update_email_args("conversion", source_args)
        )
        if tasks:
            for task in tasks:
                submit_evaluation(
                    conversion_id,
                    xp_output_dir,
                    task=task,
                    email=update_email_args("eval", source_args),
                    skip_existing_evals=skip_existing_evals,
                )
        print()
    except Exception:
        for jid in [job_id, conversion_id]:
            if jid:
                try:
                    subprocess.run(["scancel", str(jid)], check=True)
                    logger.info(f"❌ Cancelled job {jid}")
                except subprocess.CalledProcessError as cancel_err:
                    logger.warning(f"Failed to cancel job {jid}: {cancel_err}")
        logger.error(
            "💥 Job were cancelled because an error was raised when submitting the jobs!"
        )
        raise
