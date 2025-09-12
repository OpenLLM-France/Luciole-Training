import subprocess
from pathlib import Path
import argparse
import os
from slugify import slugify
import math

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output={log_dir}/eval_log_{log_name}_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
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
    "{model_arg}" \\
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
        "--hf_model",
        choices=[
            "allenai/OLMo-2-0425-1B",
            "allenai/OLMo-2-1124-7B",
            "utter-project/EuroLLM-1.7B",
            "HuggingFaceTB/SmolLM2-1.7B",
            "HuggingFaceTB/SmolLM3-3B",
            "OpenLLM-France/Lucie-7B",
        ],
        help="Use Hugging Face models.",
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
    task_to_evaluate = Path(args.task_to_evaluate)

    if args.hf_model == "allenai/OLMo-2-0425-1B":
        checkpoints = [args.hf_model for i in range(1, 20)]
        revisions = [
            f"stage1-step{i*100000}-tokens{math.ceil(i*209.73)}B" for i in range(1, 20)
        ]
        hf_ckpt_dir = Path(".")
    elif args.hf_model == "allenai/OLMo-2-1124-7B":
        checkpoints = [args.hf_model for i in range(1, 20)]
        revisions = [
            f"stage1-step{i*50000}-tokens{math.ceil(i*209.767)}B" for i in range(1, 20)
        ]
        hf_ckpt_dir = Path(".")
    elif args.hf_model == "OpenLLM-France/Lucie-7B":
        checkpoints = [args.hf_model for i in range(1, 16)]
        revisions = [f"step{i*50000:07d}" for i in range(1, 16)]
        hf_ckpt_dir = Path(".")
    elif args.hf_model in [
        "utter-project/EuroLLM-1.7B",
        "HuggingFaceTB/SmolLM2-1.7B",
        "HuggingFaceTB/SmolLM3-3B",
    ]:
        checkpoints = [args.hf_model]
        revisions = [""]
        hf_ckpt_dir = Path(".")
    else:
        hf_ckpt_dir = experiment_path / "huggingface_checkpoints"
        assert hf_ckpt_dir.is_dir(), f"Directory does not exist: {hf_ckpt_dir}"
        checkpoints = [d for d in hf_ckpt_dir.iterdir() if d.is_dir()]
        revisions = ["" for _ in checkpoints]

    # create output dirs
    output_dir = experiment_path / "evaluation" / task_to_evaluate.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / "slurm_scripts"
    job_dir.mkdir(parents=True, exist_ok=True)

    extra_arg = init_extra_args(args)

    for ckpt, revision in zip(checkpoints, revisions):
        if isinstance(ckpt, Path):
            ckpt = ckpt.name

        if (output_dir / "results" / ckpt).is_dir():
            print(f"Skipping existing results for checkpoint: {ckpt}")
            continue

        job_script = SBATCH_SCRIPT_TEMPLATE.format(
            hf_ckpt_dir=hf_ckpt_dir.resolve(),
            command=args.command,
            model_arg=f"model_name={ckpt},dtype=bfloat16"
            if not revision
            else f"model_name={ckpt},revision={revision},dtype=bfloat16",
            output_dir=output_dir if not revision else output_dir / revision,
            log_dir=log_dir,
            log_name=f"{task_to_evaluate.stem}_{slugify(ckpt)}",
            task_to_evaluate=task_to_evaluate.resolve(),
            max_samples=args.max_samples,
            extra_arg=extra_arg,
            dependency=f"#SBATCH --dependency=afterok:{args.dependency}"
            if args.dependency
            else "",
        )

        if not revision:
            job_filename = job_dir / f"job_{slugify(ckpt)}.slurm"
        else:
            job_filename = job_dir / f"job_{slugify(ckpt)}_{revision}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        print(f"Submitting job for checkpoint: {ckpt} {revision}")
        subprocess.run(["sbatch", str(job_filename)], check=True)


if __name__ == "__main__":
    main()
