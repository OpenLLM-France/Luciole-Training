import subprocess
from pathlib import Path
from argparse import ArgumentParser
import os
import re

SBATCH_CONV_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output={log_dir}/log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_{gpu}-dev
#SBATCH --account={account_gpu}
#SBATCH --constraint={gpu}
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_HUB_OFFLINE=1

module purge
module load arch/{gpu} nemo/2.4.0

torchrun --nproc_per_node=1 {source_path}/pretrain/conversion/convert_experiment.py {experiment_path} --arch {arch} --multiple_of {multiple_of}
"""

SBATCH_CONV_NEMORL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=convert_nemorl
#SBATCH --output={log_dir}/log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_{gpu}-dev
#SBATCH --account={account_gpu}
#SBATCH --constraint={gpu}
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_HUB_OFFLINE=1                                                        
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export TORCH_CUDA_ARCH_LIST=9.0

module purge
module load arch/h100
module load cuda/12.8.0
module load cudnn/9.10.2.21-12-cuda
module load nccl/2.27.3-1-cuda                          
module load uv/0.8.3         

cd $SCRATCH/nemo-rl3 #TODO make it more robust by not hardcoding nemo-rl path
source .venv/bin/activate 

python {source_path}/finetune/nemo-rl/convert_experiment.py {experiment_path} {prefix_name}
"""
SBATCH_CONV_NEMORL_LATEST_TEMPLATE ="""#!/bin/bash
#SBATCH --job-name=convert_nemorl
#SBATCH --output={log_dir}/conversion/slurm_logs/log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_{gpu}-dev
#SBATCH --account={account_gpu}
#SBATCH --constraint={gpu}
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_HUB_OFFLINE=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2


## load environment
module purge
module load arch/h100
module load uv/0.8.3
cd $SCRATCH/nemo-rl-latest
source .venv/bin/activate

# The venv's Python (miniforge 3.13) links _ssl against OpenSSL >= 3.3.0, but the
# system /lib64/libcrypto.so.3 only provides up to 3.2.0. Point at miniforge's own
# newer libcrypto so `import ssl` works (otherwise Ray's dashboard fails to start
# and GCS/raylet startup times out).
export LD_LIBRARY_PATH=/lustre/fshomisc/sup/hpe/pub/miniforge/26.1.1-3/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}

export CUDA_HOME=/lustre/fshomisc/sys/common/nvidia/cuda/12.8
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=90
export TORCH_CUDA_ARCH_LIST=9.0
python {source_path}/finetune/nemo-rl/convert_experiment.py {experiment_path} {prefix_name}
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
#SBATCH --account={account_cpu}
#SBATCH --mail-type=FAIL

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
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
eval echo "$BODY" | mailx -s "$SUBJECT" $ATTACHMENTS $TO
"""


def launch_conversion(
    experiment_path, arch, multiple_of=1, dry_run=False, is_nemo_rl=False, latest=False
):
    print(
        f"Launching conversion for {experiment_path} with arch={arch} and multiple_of={multiple_of}"
    )
    job_dir = Path(experiment_path) / "conversion"
    job_dir.mkdir(parents=True, exist_ok=True)

    account_gpu = os.environ.get("SLURM_ACCOUNT_GPU", "wuh@h100")
    gpu = account_gpu.split("@")[1]

    if not is_nemo_rl:
        job_script = SBATCH_CONV_TEMPLATE.format(
            log_dir=job_dir / "slurm_logs",
            experiment_path=experiment_path,
            arch=arch,
            multiple_of=multiple_of,
            account_gpu=account_gpu,
            gpu=gpu,
            source_path=Path(__file__).parent.parent.resolve(),
        )
    elif is_nemo_rl and not latest:
        print("Using Nemotron-RL specific conversion template")
        job_script = SBATCH_CONV_NEMORL_TEMPLATE.format(
            log_dir=job_dir / "slurm_logs",
            experiment_path=experiment_path,
            prefix_name=Path(experiment_path).name,
            account_gpu=account_gpu,
            gpu=gpu,
            source_path=Path(__file__).parent.parent.resolve(),
        )
    elif is_nemo_rl and latest:
        print("Using Nemotron-RL specific conversion template")
        job_script = SBATCH_CONV_NEMORL_LATEST_TEMPLATE.format(
            log_dir=job_dir / "slurm_logs",
            experiment_path=experiment_path,
            prefix_name=Path(experiment_path).name,
            account_gpu=account_gpu,
            gpu=gpu,
            source_path=Path(__file__).parent.parent.resolve(),
        )

    job_filename = job_dir / "conversion_job.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)

    command = ["sbatch", "--parsable", str(job_filename)]

    if dry_run:
        print(" ".join(command))
        return None

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    return job_id


def launch_evaluation(
    experiment_path,
    multiple_of=1,
    command="vllm",
    hf_model=None,
    dependency_job_id=None,
    force=False,
    eval_type="pretrain",
    lighteval_kwargs="",
    additional_model_args=None,
    infer_ckpt_name=True,
    last_checkpoint_only=False,
    dry_run=False,
):
    import evaluate_experiment

    # An eval_type of the form "base/filter" (e.g. "finetune/safety") restricts the
    # launched jobs to the tasks whose file name matches the "filter" regex.
    eval_type, _, task_filter = eval_type.partition("/")

    job_ids = []

    COMMANDS = []

    # Catch the model size from the experiment path "24B" -> 24
    matcher = re.search(r"(\d+)B", experiment_path)
    if matcher:
        model_size = int(matcher.group(1))
    else:
        matcher = re.search(r"(\d+)b", experiment_path)
        if matcher:
            model_size = int(matcher.group(1))
        else:
            model_size = 8

    gpus = 1 if model_size <= 32 else 2

    if eval_type in ["pretrain", "context_extension"]:
        COMMANDS += [
            dict(
                task_to_evaluate="tasks/gsm8k.txt",
                multiple_of=multiple_of,
                command=command,
                gpus=gpus,
            ),
            dict(
                task_to_evaluate="tasks/en.txt",
                multiple_of=multiple_of,
                command=command,
                gpus=gpus,
            ),
            dict(
                task_to_evaluate="tasks/fr.txt",
                multiple_of=multiple_of,
                command=command,
                custom_tasks="multilingual",
                max_samples=1000,
                gpus=gpus,
            ),
            dict(
                task_to_evaluate="tasks/multilingual.txt",
                multiple_of=multiple_of,
                command=command,
                custom_tasks="multilingual",
                max_samples=1000,
                gpus=gpus,
            ),
            dict(
                task_to_evaluate="tasks/mmlu_pro.txt",
                multiple_of=multiple_of,
                command=command,
                custom_tasks="smollm3",
                # max_samples=1000,
                gpus=gpus,
            ),
        ]

    if eval_type == "finetune":
        COMMANDS += (
            [
                dict(
                    task_to_evaluate=f"tasks/{task}.txt",
                    multiple_of=multiple_of,
                    command=command,
                    max_samples=1000,
                    gpus=gpus,
                )
                for task in [
                    "mixeval",
                    "ifbench",
                    "ifeval",
                    "ifeval_fr",
                    "gsm_plus",
                    "safety",
                    # "safety_multilang",
                    "rag2",
                ]
            ]
            + [
                dict(
                    task_to_evaluate=f"tasks/{task}.txt",
                    multiple_of=multiple_of,
                    command=command,
                    max_model_length=32768,
                    max_samples=1000,
                    gpus=gpus,
                )
                for task in ["aime"]
            ]
            + [
                dict(
                    task_to_evaluate=f"tasks/{task}.txt",
                    multiple_of=multiple_of,
                    command=command,
                    max_model_length=65536,
                    max_samples=1000,
                    gpus=gpus,
                )
                for task in ["live_code_bench", "gpqa", "gpqa-fr"]
            ]
            + [
                dict(
                    task_to_evaluate="tasks/mmlu_pro.txt",
                    multiple_of=multiple_of,
                    command=command,
                    custom_tasks="smollm3",
                    # max_samples=1000,
                    gpus=gpus,
                ),
                dict(
                    task_to_evaluate="tasks/reasoning.txt",
                    multiple_of=multiple_of,
                    command=command,
                    custom_tasks="multilingual",
                    max_samples=1000,
                    gpus=gpus,
                ),
                # dict(
                #     task_to_evaluate="tasks/gsm8k.txt",
                #     multiple_of=multiple_of,
                #     command=command,
                #     gpus=gpus,
                # ),
                dict(
                    task_to_evaluate="tasks/gsm8k_lighteval.txt",
                    multiple_of=multiple_of,
                    command=command,
                    gpus=gpus,
                ),
                dict(
                    task_to_evaluate="tasks/rag.txt",
                    multiple_of=multiple_of,
                    command=command,
                    custom_tasks="rag",
                    gpus=gpus,
                ),
            ]
        )

    if eval_type.startswith("ruler") or eval_type == "context_extension":
        lengths = [4096, 8192, 16384, 32768, 65536, 131072]
        if eval_type.startswith("ruler_"):
            length_str = eval_type.split("_")[1]
            lengths = [int(length_str)]

        COMMANDS += [
            dict(
                task_to_evaluate=f"tasks/ruler_{length}.txt",
                multiple_of=multiple_of,
                command=command,
                custom_tasks="ruler",
                max_model_length=int(length * 1.2),
                gpus=4
                if (length > 65536 and model_size > 12)
                else (2 if length > 32768 else 1),
            )
            for length in lengths
        ]

    if not COMMANDS:
        raise NotImplementedError(f"Unknown eval_type: {eval_type}")

    # Keep only the tasks matching the optional "base/filter" selector.
    if task_filter:
        task_regex = re.compile(task_filter)
        selected = [
            cmd
            for cmd in COMMANDS
            if task_regex.search(Path(cmd["task_to_evaluate"]).stem)
        ]
        if not selected:
            available = sorted(Path(cmd["task_to_evaluate"]).stem for cmd in COMMANDS)
            raise ValueError(
                f"Task filter '{task_filter}' matched no task of eval_type "
                f"'{eval_type}'. Available tasks: {', '.join(available)}."
            )
        COMMANDS = selected

    for command in COMMANDS:
        job_id = evaluate_experiment.launch_evaluation(
            experiment_path=experiment_path,
            **command,
            hf_model=hf_model,
            dependency=dependency_job_id,
            force=force,
            infer_ckpt_name=infer_ckpt_name,
            last_checkpoint_only=last_checkpoint_only,
            dry_run=dry_run,
            lighteval_kwargs=lighteval_kwargs,
            additional_model_args=additional_model_args,
            is_nemo_rl=args.is_nemo_rl,
        )
        if job_id is not None:
            job_ids.append(job_id)

    if job_ids:
        print(
            f"Launching evaluation for {experiment_path} with job ids: {' '.join(job_ids)}"
        )
    return ",".join(job_ids) if job_ids else None


def launch_plot(
    experiment_path, email="", dependency_job_id=None, eval_type="pretrain"
):
    print(f"Launching plot for {experiment_path}")
    # Plot groups are selected from the base eval_type, ignoring any "/filter" suffix.
    eval_type = eval_type.partition("/")[0]
    job_dir = Path(experiment_path) / "evaluation"
    job_dir.mkdir(parents=True, exist_ok=True)

    base = os.environ["OpenLLM_OUTPUT"]

    if eval_type == "finetune":
        compared_models = [
            f"{base}/finetune/compared_models/Lucie-7B-Instruct-v1.1",
            f"{base}/finetune/compared_models/SmolLM3-3B",
            f"{base}/finetune/compared_models/Ministral-3-8B-Instruct-2512",
            # f"{base}/finetune/compared_models/EuroLLM-7B-Instruct-0613",
            # f"{base}/finetune/compared_models/Gaperon-1125-7B-Instruct",
            # f"{base}/finetune/compared_models/Apertus-7B-Instruct-2509",
            # f"{base}/finetune/compared_models/salamandra-7b-instruct",
        ]

    else:  # pretrain
        if "1b" in experiment_path:
            compared_models = [
                f"{base}/pretrain/luciole_serie/luciole_nemotron1b",
                f"{base}/pretrain/luciole_serie/luciole_variant_nemotron1b_phase2",
                f"{base}/pretrain/luciole_serie/luciole_nemotron1b_annealin",
                f"{base}/pretrain/luciole_serie/luciole_32k_nemotron1b_context_extension",
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

    plot_groups = "all"
    if eval_type.startswith("ruler"):
        plot_groups = "ruler"
    elif eval_type == "pretrain":
        plot_groups = "en fr multilingual"
    elif eval_type == "finetune":
        plot_groups = "finetune"
    elif eval_type == "context_extension":
        plot_groups = "ruler en fr multilingual"

    job_script = SBATCH_PLOT_TEMPLATE.format(
        log_dir=job_dir / "slurm_logs",
        experiment_path=experiment_path,
        compared_models=" ".join(compared_models),
        plot_groups=plot_groups,
        email=email.replace(",", " "),
        account_cpu=os.environ.get("SLURM_ACCOUNT_CPU", "qgz@cpu"),
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
        help=(
            "Type of evaluation to perform: one of 'pretrain', 'finetune', 'ruler', "
            "'ruler_<length>' (e.g. 'ruler_4096'), 'context_extension'. "
            "Append '/<filter>' to launch only the tasks whose file name matches the "
            "<filter> regex, e.g. 'finetune/safety' (safety.txt + safety_fr.txt) or "
            "'finetune/safety_fr' (safety_fr.txt only)."
        ),
    )
    parser.add_argument(
        "--command",
        type=str,
        default="vllm",
        choices=["vllm", "accelerate"],
        help="Command to use for evaluation.",
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
    parser.add_argument(
        "--last_checkpoint_only",
        action="store_true",
        help="If set, only evaluate the last checkpoint.",
    )
    parser.add_argument(
        "--lighteval_kwargs",
        type=str,
        default="--save-details",
        help="Additional arguments to pass to lighteval (default: --lighteval_kwargs=--save-details).",
    )
    parser.add_argument(
        "--additional_model_args",
        type=str,
        default=None,
        help="Additional model args to pass to lighteval, separated by commas (format 'arg1=value1,arg2=value2', e.g. --additional_model_args='override_chat_template=False').",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="If set, do not submit jobs."
    )
    parser.add_argument(
        "--is_nemo_rl", action="store_true", help="Convert Nemo-RL ckpts."
    )
    parser.add_argument(
        "--latest", action="store_true", help="Convert Nemo-RL ckpts."
    )
    args = parser.parse_args()

    if not args.is_nemo_rl:
        has_original_checkpoints = (
            Path(args.experiment_path)
            / args.experiment_path.rstrip("/").split("/")[-1]
            / "checkpoints"
        ).is_dir()
    else:
        has_original_checkpoints = (Path(args.experiment_path) / "checkpoints").is_dir()
    launch_conversion_needed = not args.hf_model and has_original_checkpoints

    # Launch conversion job
    if launch_conversion_needed:
        conversion_job_id = launch_conversion(
            args.experiment_path,
            args.arch,
            args.multiple_of,
            dry_run=args.dry_run,
            is_nemo_rl=args.is_nemo_rl,
            latest=args.latest
        )
        if args.dry_run:
            print("#" * 80)
    else:
        conversion_job_id = None

    # Launch evaluation job
    evaluation_job_id = launch_evaluation(
        args.experiment_path,
        args.multiple_of,
        hf_model=args.hf_model,
        command=args.command,
        dependency_job_id=conversion_job_id,
        force=args.force,
        eval_type=args.eval_type,
        infer_ckpt_name=launch_conversion_needed,
        lighteval_kwargs=args.lighteval_kwargs,
        additional_model_args=args.additional_model_args,
        last_checkpoint_only=args.last_checkpoint_only,
        dry_run=args.dry_run,
    )

    if not args.hf_model and evaluation_job_id:
        # Plot
        launch_plot(
            args.experiment_path,
            dependency_job_id=evaluation_job_id,
            email=args.email,
            eval_type=args.eval_type,
        )
