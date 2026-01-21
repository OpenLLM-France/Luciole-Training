import subprocess
from pathlib import Path
from argparse import ArgumentParser
import os

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
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load arch/h100 nemo/2.4.0

torchrun --nproc_per_node=1 ../conversion/convert_experiment.py {experiment_path} --arch {arch} --multiple_of {multiple_of}
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
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load anaconda-py3/2024.06
conda activate eval-env

python plot_results.py {compared_models} --group all {plot_groups} --output_path {experiment_path}/figs --dpi 150

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
    # Skip files starting with "all"
    filename=$(basename "$f")
    [[ "$filename" == all* ]] && continue

    ATTACHMENTS="$ATTACHMENTS -a \"$f\""
done

if [[ -z "$ATTACHMENTS" ]]; then
    echo "Warning: no files to attach in $FOLDER."
fi

# Send email
eval echo "$BODY" | mailx -s "$SUBJECT" $ATTACHMENTS "$TO"
"""


def launch_conversion(experiment_path, arch, multiple_of=1):
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

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    return job_id


def launch_evaluation(
    experiment_path, multiple_of=1, hf_model=None, dependency_job_id=None, force=False, eval_type="pretrain"
):
    import evaluate_experiment

    job_ids = []

    COMMANDS_PRETRAIN = [
        dict(
            task_to_evaluate="tasks/gsm8k.txt", multiple_of=multiple_of, command="vllm"
        ),
        dict(task_to_evaluate="tasks/en.txt", multiple_of=multiple_of, command="vllm"),
        dict(
            task_to_evaluate="tasks/fr.txt",
            multiple_of=multiple_of,
            command="vllm",
            custom_tasks="multilingual",
            max_samples=1000,
        ),
        dict(
            task_to_evaluate="tasks/multilingual.txt",
            multiple_of=multiple_of,
            command="vllm",
            custom_tasks="multilingual",
            max_samples=1000,
        ),
    ]

    if eval_type == "pretrain":
        COMMANDS = COMMANDS_PRETRAIN
        
    elif eval_type == "finetune":
        COMMANDS = [
            dict(
                task_to_evaluate=f"tasks/{task}.txt",
                multiple_of=multiple_of,
                command="vllm",
                max_samples=1000,
            )
            for task in ["mixeval", "ifbench", "ifeval", "ifeval_fr", "gsm_plus"]
        ] + [
            dict(
                task_to_evaluate=f"tasks/{task}.txt",
                multiple_of=multiple_of,
                command="vllm",
                max_model_length=32768,
                max_samples=1000,
            )
            for task in ["aime"]
        ] + [
            dict(
                task_to_evaluate=f"tasks/{task}.txt",
                multiple_of=multiple_of,
                command="vllm",
                max_model_length=65536,
                max_samples=1000,
            )
            for task in ["live_code_bench", "gpqa", "gpqa-fr"]
        ] + [
            dict(
                task_to_evaluate="tasks/mmlu_pro.txt",
                multiple_of=multiple_of,
                command="vllm",
                custom_tasks="smollm3",
                max_samples=1000,
            ),
            dict(
                task_to_evaluate="tasks/reasoning.txt",
                multiple_of=multiple_of,
                command="vllm",
                custom_tasks="multilingual",
                max_samples=1000,
            ),
            dict(
                task_to_evaluate="tasks/gsm8k.txt", multiple_of=multiple_of, command="vllm"
            ),
        ]
    elif eval_type.startswith("ruler"):

        lengths = [4096, 8192, 16384, 32768, 65536, 131072]
        if eval_type.startswith("ruler_"):
            length_str = eval_type.split("_")[1]
            lengths = [int(length_str)]

        COMMANDS = [
            dict(
                task_to_evaluate=f"tasks/ruler_{length}.txt",
                multiple_of=multiple_of,
                command="vllm",
                custom_tasks="ruler",
                max_model_length=length,
                gpus=2 if length > 32768 else 1,
                
            ) for length in lengths
        ]
    else:
        raise NotImplementedError(f"Unknown eval_type: {eval_type}")
    for command in COMMANDS:
        job_id = evaluate_experiment.launch_evaluation(
            experiment_path=experiment_path,
            **command,
            hf_model=hf_model,
            dependency=dependency_job_id,
            force=force,
            infer_ckpt_name=True,
        )
        if job_id is not None:
            job_ids.append(job_id)

    print(
        f"Launching evaluation for {experiment_path} with job ids: {','.join(job_ids)}"
    )
    return ",".join(job_ids) if job_ids else None


def launch_plot(experiment_path, email="", dependency_job_id=None, eval_type="pretrain"):
    print(f"Launching plot for {experiment_path}")
    job_dir = Path(experiment_path) / "evaluation"
    job_dir.mkdir(parents=True, exist_ok=True)

    base = os.environ["OpenLLM_OUTPUT"]

    if eval_type == "finetune":

        compared_models = [
            f"{base}/finetune/sftmix_8b_lrablations.eval",
            f"{base}/finetune/compared_models/Lucie-7B-Instruct-v1.1",
            f"{base}/finetune/compared_models/SmolLM3-3B",
            # f"{base}/finetune/compared_models/EuroLLM-7B-Instruct-0613",
            # f"{base}/finetune/compared_models/Gaperon-1125-7B-Instruct",
            # f"{base}/finetune/compared_models/Apertus-7B-Instruct-2509",
            # f"{base}/finetune/compared_models/salamandra-7b-instruct",
        ]

    else: # pretrain
        if "1b" in experiment_path:
            compared_models = [
                f"{base}/pretrain/luciole_serie/luciole_nemotron1b",
                f"{base}/pretrain/luciole_serie/luciole_variant_nemotron1b_phase2",
                f"{base}/pretrain/luciole_serie/luciole_nemotron1b_annealin",
                f"{base}/pretrain/luciole_serie/luciole_32k_nemotron1b_annealing",
                f"{base}/pretrain/compared_models/OLMo-2-0425-1B",
                f"{base}/pretrain/compared_models/EuroLLM-1.7B",
                f"{base}/pretrain/compared_models/Gaperon-1125-1B",
                f"{base}/pretrain/compared_models/CroissantLLMBase",
            ]
        elif "8b" in experiment_path:
            compared_models = [
                f"{base}/pretrain/luciole_serie/luciole_nemotronh8b_phase1",
                f"{base}/pretrain/luciole_serie/luciole_nemotronh8b_phase2",
                f"{base}/pretrain/compared_models/OLMo-2-1124-7B",
                f"{base}/pretrain/compared_models/EuroLLM-9B",
                f"{base}/pretrain/compared_models/Gaperon-1125-8B",
                f"{base}/pretrain/compared_models/Apertus-8B-2509",
                f"{base}/pretrain/compared_models/salamandra-7b",
                f"{base}/pretrain/compared_models/Lucie-7B",
            ]
        elif "23b" in experiment_path:
            compared_models = [
                f"{base}/pretrain/luciole_serie/luciolr_nemotron23b_phase1",
                f"{base}/pretrain/luciole_serie/luciolr_lower_nemotron23b_phase1",
                f"{base}/pretrain/luciole_serie/luciole_nemotron23b_phase2",
                f"{base}/pretrain/compared_models/OLMo-2-0325-13B",
                f"{base}/pretrain/compared_models/OLMo-2-0325-32B",
                f"{base}/pretrain/compared_models/Gaperon-1125-24B",
            ]
        else:
            compared_models = []
    if experiment_path not in compared_models:
        compared_models = [experiment_path] + compared_models

    if eval_type.startswith("ruler"):
        plot_groups = "ruler"
    elif eval_type == "pretrain":
        plot_groups = "en fr multilingual"
    elif eval_type == "finetune":
        plot_groups = "finetune"

    job_script = SBATCH_PLOT_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs",
        experiment_path=experiment_path,
        compared_models=" ".join(compared_models),
        plot_groups=plot_groups,
        email=email,
    )

    job_filename = job_dir / "plot_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    command = ["sbatch", str(job_filename)]
    if dependency_job_id:
        command.insert(1, f"--dependency=afterany:{dependency_job_id}")

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=str,
        default=None,
        help="Path to an experiment",
    )
    parser.add_argument(
        "--hf_model",
        default=None,
        help="Use Hugging Face models.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="nemotron",
        choices=["llama", "nemotron", "nemotronh"],
    )
    parser.add_argument(
        "--multiple_of",
        type=int,
        default=1,
        help="Convert and evaluate only checkpoints that are multiple of this value",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune", "ruler", "ruler_4096"],
        help="Type of evaluation to perform.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="Email to send final figures (use spacing if you have more than one recipient)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, force re-evaluation even if results exist.",
    )
    args = parser.parse_args()

    # Launch conversion job
    if not args.hf_model:
        conversion_job_id = launch_conversion(
            args.experiment_path,
            args.arch,
            args.multiple_of,
        )
    else:
        conversion_job_id = None

    # Launch evaluation job
    evaluation_job_id = launch_evaluation(
        args.experiment_path,
        args.multiple_of,
        hf_model=args.hf_model,
        dependency_job_id=conversion_job_id,
        force=args.force,
        eval_type=args.eval_type,
    )

    # Plot
    launch_plot(
        args.experiment_path, dependency_job_id=evaluation_job_id, email=args.email, eval_type=args.eval_type
    )
