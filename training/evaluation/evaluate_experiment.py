import subprocess
from pathlib import Path
import argparse
import os

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output={log_dir}/eval_log_{log_name}_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{dependency}

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

cd {hf_ckpt_dir}

mkdir -p {output_dir}

lighteval {command} \\
    "model_name={model_name},dtype=bfloat16" \\
    "{task_to_evaluate}" \\
    --output-dir {output_dir} \\
    {extra_arg}
"""


def init_extra_args(args):
    extra_arg = ""
    # Custom tasks
    if args.custom_tasks is None:
        pass
    elif args.custom_tasks == "multilingual":
        extra_arg += "--custom-tasks lighteval.tasks.multilingual.tasks \\"
    elif args.custom_tasks == "fineweb":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "fineweb_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    elif args.custom_tasks == "lucie":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "lucie2_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    else:
        raise ValueError(f"Unknown custom_tasks: {args.custom_tasks}")
    # Max samples
    if args.max_samples > 0:
        extra_arg += f"--max-samples {args.max_samples} \\"
    return extra_arg


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for each model checkpoint."
    )
    parser.add_argument(
        "experiment_path", type=str, help="Path to the experiment directory."
    )
    parser.add_argument(
        "task_to_evaluate", type=str, help="Path to the task txt file or task name."
    )
    parser.add_argument(
        "--command", type=str, default="vllm", choices=["vllm", "accelerate"], help=""
    )
    parser.add_argument(
        "--custom_tasks",
        default=None,
        choices=[None, "multilingual", "fineweb", "lucie"],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        help="A dependency after which it should launch the evals",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    hf_ckpt_dir = experiment_path / "huggingface_checkpoints"
    assert hf_ckpt_dir.is_dir(), f"Directory does not exist: {hf_ckpt_dir}"
    task_to_evaluate = Path(args.task_to_evaluate)
    # create output dirs
    output_dir = experiment_path / "evaluation" / task_to_evaluate.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / "slurm_scripts"
    job_dir.mkdir(parents=True, exist_ok=True)

    extra_arg = init_extra_args(args)

    checkpoints = [d for d in hf_ckpt_dir.iterdir() if d.is_dir()]

    for ckpt in checkpoints:
        ckpt_name = ckpt.name  # ckpt_name = ckpt.name.replace("=", "_")  # escape '=' just in case - Deprecated

        if (output_dir / "results" / ckpt_name).is_dir():
            print(f"Skipping existing results for checkpoint: {ckpt_name}")
            continue

        job_script = SBATCH_SCRIPT_TEMPLATE.format(
            hf_ckpt_dir=hf_ckpt_dir.resolve(),
            command=args.command,
            model_name=ckpt_name,
            output_dir=output_dir,
            log_dir=log_dir,
            log_name=f"{task_to_evaluate.stem}_{ckpt_name}",
            task_to_evaluate=task_to_evaluate.resolve(),
            max_samples=args.max_samples,
            extra_arg=extra_arg,
            dependency=f"#SBATCH --dependency=afterok:{args.dependency}"
            if args.dependency
            else "",
        )

        job_filename = job_dir / f"job_{ckpt_name}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        print(f"Submitting job for checkpoint: {ckpt_name}")
        subprocess.run(["sbatch", str(job_filename)], check=True)


if __name__ == "__main__":
    main()
