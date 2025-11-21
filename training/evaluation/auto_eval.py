import subprocess
from pathlib import Path

SBATCH_CONV_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output={log_dir}/log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load arch/h100 nemo/2.4.0

torchrun --nproc_per_node=1 ../conversion/convert_experiment.py {experiment_path} --arch {arch} --multiple_of {multiple_of}
"""

SBATCH_EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output={log_dir}/log_%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

echo {str_args}
python evaluate_experiment.py {str_args}
"""


def launch_conversion(args, slurm_args=None):
    experiment_path = args.experiment_path
    arch = args.arch
    multiple_of = args.multiple_of

    print(
        f"Launching conversion for {experiment_path} with arch={arch} and multiple_of={multiple_of}"
    )
    job_dir = Path(experiment_path) / "conversion"
    job_dir.mkdir(parents=True, exist_ok=True)

    job_script = SBATCH_CONV_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs",
        experiment_path=experiment_path,
        arch=arch,
        multiple_of=multiple_of,
    )

    job_filename = job_dir / "conversion_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    command = ["sbatch", "--parsable", str(job_filename)]

    if slurm_args:
        slurm_args = vars(slurm_args)
        for key, value in slurm_args.items():
            command.insert(1, f"--{key}={value}")

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    return job_id


def dict_to_cli(args_dict, positional_args=["experiment_path", "task_to_evaluate"]):
    cli_parts = []
    for k, v in args_dict.items():
        if k in positional_args:
            cli_parts.append(str(v))
        elif isinstance(v, bool):
            if v:  # only include True flags
                cli_parts.append(f"\\\n    --{k}")
        elif isinstance(v, str):
            cli_parts.append(f"\\\n    --{k} '{v}'")
        elif v is not None:
            cli_parts.append(f"\\\n    --{k} {v}")
    return " ".join(cli_parts)


def launch_evaluation(args, dependency_job_id=None):
    experiment_path = args.experiment_path
    multiple_of = args.multiple_of

    print(f"Launching evaluation for {experiment_path} and multiple_of={multiple_of}")
    job_dir = Path(experiment_path) / "evaluation"
    job_dir.mkdir(parents=True, exist_ok=True)

    str_args = dict_to_cli(vars(args))
    job_script = SBATCH_EVAL_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs", str_args=str_args
    )

    job_filename = job_dir / "evaluation_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    subprocess.run(
        ["sbatch", f"--dependency=afterok:{dependency_job_id}", str(job_filename)],
        check=True,
    )


def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--begin",
        type=str,
        default=None,
        help="Begin time for slurm job (e.g., now+30minutes)",
    )
    return parser


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from conversion.convert_experiment import get_parser as get_conv_parser
    from evaluate_experiment import get_parser as get_eval_parser

    parser = get_parser()
    conv_parser = get_conv_parser()
    eval_parser = get_eval_parser()

    # Parse each independently
    conv_args, _ = conv_parser.parse_known_args()
    eval_args, _ = eval_parser.parse_known_args()
    slurm_args, _ = parser.parse_known_args()

    # Launch conversion job
    conversion_job_id = launch_conversion(conv_args, slurm_args)

    # Launch evaluation job
    launch_evaluation(eval_args, dependency_job_id=conversion_job_id)
