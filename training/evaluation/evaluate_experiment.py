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
        "--multilingual",
        action="store_true",
        help="Use multilingual task configuration.",
    )
    parser.add_argument(
        "--fineweb",
        action="store_true",
        help="Use fineweb2 task configuration.",
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
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip evaluation if output already exists.",
        default=False,
    )
    args = parser.parse_args()

    assert not (
        args.multilingual and args.fineweb
    ), "Both --multilingual and --fineweb cannot be activated simultaneously"

    experiment_path = Path(args.experiment_path)
    hf_ckpt_dir = experiment_path / "huggingface_checkpoints"
    output_dir = experiment_path / "evaluation"
    assert hf_ckpt_dir.is_dir(), f"Directory does not exist: {hf_ckpt_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_to_evaluate = Path(args.task_to_evaluate)

    checkpoints = [d for d in hf_ckpt_dir.iterdir() if d.is_dir()]

    extra_arg = ""
    if args.multilingual:
        extra_arg += "--custom-tasks lighteval.tasks.multilingual.tasks \\"
    if args.fineweb:
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "fineweb_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    if args.max_samples > 0:
        extra_arg += f"--max-samples {args.max_samples} \\"

    skipped_all = True
    for ckpt in checkpoints:
        ckpt_name = ckpt.name.replace("=", "_")  # escape '=' just in case
        log_dir = output_dir / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing:
            completed_path = output_dir / "results" / ckpt_name
            if os.path.exists(completed_path):
                completed_files = os.listdir(completed_path)
                if any(f.startswith("results_") and f.endswith(".json") for f in completed_files):
                    print(f"Skipping evaluation for {ckpt_name} of {os.path.basename(experiment_path)} as results already exist ({completed_files}).")
                    continue
        skipped_all = False
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

        job_filename = output_dir / f"job_{ckpt_name}_{task_to_evaluate.stem}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        print(f"Submitting job for checkpoint: {ckpt_name}")
        subprocess.run(["sbatch", str(job_filename)], check=True)
    if args.skip_existing and skipped_all:
        completed_file = output_dir / "completed.txt"
        with open(completed_file, "w") as f:
            f.write("")

if __name__ == "__main__":
    main()
