import subprocess
from pathlib import Path
import argparse
import os
from slugify import slugify
import math
import re

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output={log_dir}/eval_log_{log_name}_%j.out
#SBATCH --gres=gpu:{gpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{dependency}

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
module load nccl/2.27.3-1-cuda
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

cd {ckpt_dir}

mkdir -p {output_dir}

VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval {command} \\
    "{model_arg}" \\
    "{task_to_evaluate}" \\
    --output-dir {output_dir} \\
    {extra_arg}
"""


def init_extra_args(custom_tasks, max_samples=-1):
    extra_arg = ""
    # Custom tasks
    if custom_tasks is None:
        pass
    elif custom_tasks == "multilingual":
        extra_arg += "--custom-tasks lighteval.tasks.multilingual.tasks \\"
    elif custom_tasks == "smollm3":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "smollm3_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    elif custom_tasks == "fineweb":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "fineweb_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    elif custom_tasks == "lucie":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "lucie2_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\"
    else:
        raise ValueError(f"Unknown custom_tasks: {custom_tasks}")
    # Max samples
    if max_samples > 0:
        extra_arg += f"--max-samples {max_samples} \\"
    return extra_arg


def get_hf_model(hf_model):
    if hf_model == "allenai/OLMo-2-0425-1B":
        checkpoints = [hf_model for i in range(1, 20)]
        revisions = [
            f"stage1-step{i*100000}-tokens{math.ceil(i*209.73)}B" for i in range(1, 20)
        ]
    elif hf_model == "allenai/OLMo-2-1124-7B":
        checkpoints = [hf_model for i in range(1, 20)]
        revisions = [
            f"stage1-step{i*50000}-tokens{math.ceil(i*209.767)}B" for i in range(1, 20)
        ]
    elif hf_model == "OpenLLM-France/Lucie-7B":
        checkpoints = [hf_model for i in range(1, 16)]
        revisions = [f"step{i*50000:07d}" for i in range(1, 16)]
    elif hf_model == "HuggingFaceTB/SmolLM2-1.7B":
        checkpoints = [
            "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints" for i in range(1, 20)
        ] + ["HuggingFaceTB/SmolLM2-1.7B"]
        revisions = [f"step-{i*250000}" for i in range(1, 20)] + ["main"]
    else:
        print(f"Selection the main revision of {hf_model} model.")
        checkpoints = [hf_model]
        revisions = ["main"]
    return checkpoints, revisions


def get_checkpoints_and_revisions(experiment_path, hf_model=None):
    if hf_model is not None:
        checkpoints, revisions = get_hf_model(hf_model)
        ckpt_dir = Path(".")
    else:
        ckpt_dir = experiment_path / "huggingface_checkpoints"
        assert ckpt_dir.is_dir(), f"Directory does not exist: {ckpt_dir}"
        checkpoints = [d for d in ckpt_dir.iterdir() if d.is_dir()]
        revisions = ["" for _ in checkpoints]
    return checkpoints, revisions, ckpt_dir

def get_step(text):
    match = re.search(r"-step[=_](\d+)", text)
    if match:
        step_number = int(match.group(1))
        return step_number
    else:
        return None

def launch_evaluation(
    experiment_path,
    task_to_evaluate,
    hf_model,
    custom_tasks,
    evaluation_dir,
    command,
    max_samples=-1,
    dependency=None,
    lighteval_kwargs="",
    force=False,
    debug=False,
    multiple_of=None,
    gpus=1,
):
    experiment_path = Path(experiment_path)
    task_to_evaluate = Path(task_to_evaluate)
    print(f"\nExperiment path: {experiment_path}")
    print(f"Task to evaluate: {task_to_evaluate}")

    checkpoints, revisions, ckpt_dir = get_checkpoints_and_revisions(
        experiment_path, hf_model
    )

    # create output dirs
    output_dir = experiment_path / evaluation_dir / task_to_evaluate.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / "slurm_scripts"
    job_dir.mkdir(parents=True, exist_ok=True)

    extra_arg = init_extra_args(custom_tasks, max_samples)
    extra_arg += lighteval_kwargs

    for ckpt, revision in zip(checkpoints, revisions):
        if isinstance(ckpt, Path):
            ckpt = ckpt.name
        
        if multiple_of:
            step = get_step(ckpt)
            if ((step + 1) % args.multiple_of != 0):
                print(f"Skipping checkpoint: {ckpt} {revision}. Step {step + 1} is not a multiple of {args.multiple_of}")
                continue

        if (output_dir / "results" / ckpt).is_dir() and not force:
            print(f"Skipping existing results for checkpoint: {ckpt}")
            continue

        model_arg = f"model_name={ckpt},dtype=bfloat16"
        if "nemotronh" in ckpt:
            model_arg += ",trust_remote_code=True"
            if command == "vllm":
                model_arg += ",max_num_batched_tokens=4096,max_num_seqs=1"
        if revision:
            model_arg += f",revision={revision}"

        job_script = SBATCH_SCRIPT_TEMPLATE.format(
            ckpt_dir=ckpt_dir.resolve(),
            command=command,
            model_arg=model_arg,
            output_dir=output_dir if not revision else output_dir / revision,
            log_dir=log_dir,
            log_name=f"{task_to_evaluate.stem}_{slugify(ckpt)}",
            task_to_evaluate=task_to_evaluate.resolve(),
            extra_arg=extra_arg,
            gpus=gpus,
            dependency=f"#SBATCH --dependency=afterok:{dependency}"
            if dependency
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

        if debug:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for each model checkpoint."
    )
    parser.add_argument(
        "experiment_path", type=str, help="Path to the experiment directory."
    )
    parser.add_argument(
        "--hf_model",
        default=None,
        help="Use Hugging Face models.",
    )
    parser.add_argument(
        "task_to_evaluate", type=str, help="Path to the task txt file or task name."
    )
    parser.add_argument(
        "--command",
        type=str,
        default="accelerate",
        choices=["vllm", "accelerate"],
        help="",
    )
    parser.add_argument(
        "--custom_tasks",
        default=None,
        choices=[None, "multilingual"],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        help="A dependency after which it should launch the evals",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lighteval_kwargs", type=str, default="")
    parser.add_argument("--multiple_of", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    launch_evaluation(
        experiment_path=args.experiment_path,
        task_to_evaluate=args.task_to_evaluate,
        hf_model=args.hf_model,
        evaluation_dir=args.evaluation_dir,
        custom_tasks=args.custom_tasks,
        command=args.command,
        max_samples=args.max_samples,
        dependency=args.dependency,
        lighteval_kwargs=args.lighteval_kwargs,
        force=args.force,
        debug=args.debug,
        multiple_of=args.multiple_of,
        gpus=args.gpus
    )
