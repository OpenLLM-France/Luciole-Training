import os
import re
import subprocess
from pathlib import Path
from slurm_launcher import create_parser, pre_submit, write_launch_slurm, logger


def create_slurm_script(job_id, xp_output_dir):
    job_name = f"convertion_of_{job_id}"
    set_env_path = Path(__file__).resolve().parent.parent
    set_env_path = f"{set_env_path}/set_env.sh"
    experiment_dir = xp_output_dir
    convert_script_path = Path(__file__).resolve().parent.parent
    convert_script_path = f"{convert_script_path}/conversion"

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
#SBATCH --dependency=afterok:{job_id}

source {set_env_path}

echo $1
python -m torch.distributed.launch --nproc_per_node 1 {convert_script_path}/convert_experiment.py {experiment_dir}
"""
    return script


def submit_conversion(job_id, xp_output_dir):
    slurm_script = create_slurm_script(job_id, xp_output_dir)
    sbatch_script_path = os.path.join(xp_output_dir, "conversion/convertion.slurm")
    os.makedirs(os.path.join(xp_output_dir, "conversion"), exist_ok=True)
    job_id = write_launch_slurm(sbatch_script_path, slurm_script)
    return job_id


def submit_evaluation(job_id, xp_output_dir, task="en.txt"):
    path_to_evaluation = Path(__file__).resolve().parent.parent
    path_to_evaluation = f"{path_to_evaluation}/evaluation"
    language = "" if task == "en" else "multilingual"

    sbatch_script = os.path.join(path_to_evaluation, "evaluate_experiment.slurm")
    task_path = os.path.join(path_to_evaluation, "tasks", task)
    logger.info(
        f"Submitting evaluation with task: {task} and experiment: {xp_output_dir}"
    )
    result = subprocess.run(
        [
            "sbatch",
            f"--dependency=afterok:{job_id}",
            "--job-name=eval",
            f"--output={os.path.join(xp_output_dir, 'evaluation', 'log_%x')}",
            sbatch_script,
            xp_output_dir,
            task_path,
            language,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = int(match.group(1))
    logger.info(f"Evaluation submitted {job_id}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    job_id, xp_output_dir = pre_submit(args)
    job_id = submit_conversion(job_id, xp_output_dir)
    submit_evaluation(job_id, xp_output_dir, task="en.txt")
    submit_evaluation(job_id, xp_output_dir, task="fr.txt")
