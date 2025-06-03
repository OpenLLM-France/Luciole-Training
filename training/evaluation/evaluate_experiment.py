import subprocess
from pathlib import Path
import argparse

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output={log_dir}/eval_log_%j.out
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

lighteval vllm \\
    "pretrained={model_name},dtype=bfloat16" \\
    "{task_to_evaluate}" \\
    --output-dir {output_dir} \\
    --max-samples 1000 \\
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
        "--multilingual",
        action="store_true",
        help="Use multilingual task configuration.",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        help="A dependency after which it should launch the evals",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    hf_ckpt_dir = experiment_path / "huggingface_checkpoints"
    output_dir = experiment_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_to_evaluate = Path(args.task_to_evaluate)

    checkpoints = [d for d in hf_ckpt_dir.iterdir() if d.is_dir()]

    for ckpt in checkpoints:
        ckpt_name = ckpt.name.replace("=", "_")  # escape '=' just in case
        log_dir = output_dir / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        job_script = SBATCH_SCRIPT_TEMPLATE.format(
            hf_ckpt_dir=hf_ckpt_dir.resolve(),
            model_name=ckpt_name,
            output_dir=output_dir,
            log_dir=log_dir,
            task_to_evaluate=task_to_evaluate.resolve(),
            extra_arg="--custom-tasks lighteval.tasks.multilingual.tasks"
            if args.multilingual
            else "",
            dependency=f"#SBATCH --dependency=afterok:{args.dependency}"
            if args.dependency
            else "",
        )

        job_filename = output_dir / f"job_{ckpt_name}_{task_to_evaluate.stem}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        print(f"Submitting job for checkpoint: {ckpt_name}")
        subprocess.run(["sbatch", str(job_filename)], check=True)


if __name__ == "__main__":
    main()
