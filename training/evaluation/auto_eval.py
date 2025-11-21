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

python evaluate_experiment.py {experiment_path} tasks/gsm8k.txt --multiple_of {multiple_of} --command vllm 
python evaluate_experiment.py {experiment_path} tasks/en.txt --multiple_of {multiple_of} --command vllm
python evaluate_experiment.py {experiment_path} tasks/fr.txt --multiple_of {multiple_of} --command vllm --custom_tasks multilingual --max_samples 1000
python evaluate_experiment.py {experiment_path} tasks/multilingual.txt --multiple_of {multiple_of} --command vllm --custom_tasks multilingual --max_samples 1000
"""

SBATCH_PLOT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output={log_dir}/log_%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu
#SBATCH --partition=prepost

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load anaconda-py3/2024.06
conda activate eval-env

python plot_results.py {experiment_path} --group all en fr multilingual --output_path {experiment_path}/figs

# Variables
TO="{email}"
EXPERIMENT_PATH="{experiment_path}"
EXPERIMENT_NAME=$(basename "$EXPERIMENT_PATH")
SUBJECT="Evaluation $EXPERIMENT_NAME"
BODY="Please find the attachments."
FOLDER="{experiment_path}/figs"

# Check that TO is not empty
if [[ -z "$TO" ]]; then
    echo "Error: TO is empty, aborting."
    exit 1
fi

# Build the attachment parameters for mailx
ATTACHMENTS=""
for f in "$FOLDER"/*; do
    # Skip if folder is empty or no files match
    [[ -e "$f" ]] || continue
    ATTACHMENTS="$ATTACHMENTS -a \"$f\""
done

if [[ -z "$ATTACHMENTS" ]]; then
    echo "Warning: no files to attach in $FOLDER."
fi

# Send email
eval echo "$BODY" | mailx -s "$SUBJECT" $ATTACHMENTS "$TO"
"""


def launch_conversion(experiment_path, arch, multiple_of=1, begin=None):
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

    if begin:
        command.insert(1, f"--begin={begin}")

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    return job_id


def launch_evaluation(experiment_path, multiple_of=1, dependency_job_id=None):
    print(f"Launching evaluation for {experiment_path}")
    job_dir = Path(experiment_path) / "evaluation"
    job_dir.mkdir(parents=True, exist_ok=True)

    job_script = SBATCH_EVAL_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs",
        experiment_path=experiment_path,
        multiple_of=multiple_of,
    )

    job_filename = job_dir / "evaluation_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    command = ["sbatch", "--parsable", str(job_filename)]
    command.insert(1, f"--dependency=afterok:{dependency_job_id}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    return job_id


def launch_plot(experiment_path, email="", dependency_job_id=None):
    print(f"Launching plot for {experiment_path}")
    job_dir = Path(experiment_path) / "evaluation"
    job_dir.mkdir(parents=True, exist_ok=True)

    job_script = SBATCH_PLOT_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs", experiment_path=experiment_path, email=email
    )

    job_filename = job_dir / "plot_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    subprocess.run(
        ["sbatch", f"--dependency=afterok:{dependency_job_id}", str(job_filename)],
        check=True,
    )


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from conversion.convert_experiment import get_parser as get_conv_parser

    parser = get_conv_parser()
    parser.add_argument(
        "--begin",
        type=str,
        default=None,
        help="Begin time for slurm job (e.g., now+30minutes)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="Email to send final figures",
    )
    args = parser.parse_args()

    # Launch conversion job
    conversion_job_id = launch_conversion(
        args.experiment_path, args.arch, args.multiple_of, begin=args.begin
    )

    # Launch evaluation job
    evaluation_job_id = launch_evaluation(
        args.experiment_path, args.multiple_of, dependency_job_id=conversion_job_id
    )

    # Plot
    launch_plot(
        args.experiment_path, dependency_job_id=evaluation_job_id, email=args.email
    )
