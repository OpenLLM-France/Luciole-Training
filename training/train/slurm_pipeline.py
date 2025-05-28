import os
from pathlib import Path
from slurm_launcher import create_parser, pre_submit, write_launch_slurm


def create_slurm_conversion_script(job_id, xp_output_dir):
    job_name = f"conversion_of_{job_id}"
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
    slurm_script = create_slurm_conversion_script(job_id, xp_output_dir)
    sbatch_script_path = os.path.join(xp_output_dir, "conversion/conversion.slurm")
    os.makedirs(os.path.join(xp_output_dir, "conversion"), exist_ok=True)
    job_id = write_launch_slurm(sbatch_script_path, slurm_script)
    return job_id


def create_slurm_eval_script(job_id, xp_output_dir, task):
    job_name = f"evaluation_of_{job_id}"
    experiment_dir = xp_output_dir
    path_to_evaluation = Path(__file__).resolve().parent.parent
    path_to_evaluation = f"{path_to_evaluation}/evaluation"
    task_path = os.path.join(path_to_evaluation, "tasks", task)
    multilingual = False if task == "en" else True

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={xp_output_dir}/evaluation/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu
#SBATCH --dependency=afterok:{job_id}

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
    job_id, xp_output_dir = pre_submit(args)
    job_id = submit_conversion(job_id, xp_output_dir)
    submit_evaluation(job_id, xp_output_dir, task="en.txt")
    submit_evaluation(job_id, xp_output_dir, task="fr.txt")
