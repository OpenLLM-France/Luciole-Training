"""Evaluate the checkpoints of an experiment on BFCL.

BFCL (Berkeley Function-Calling Leaderboard) lives outside lighteval: it ships
its own CLI (``bfcl generate`` / ``bfcl evaluate``) and its own conda env, so it
gets its own launcher instead of being a task file of
``evaluate_experiment.py``. See the "Tool Evaluation" section of README.md for
the installation.

One SLURM array task per checkpoint; everything is written under
``<experiment_path>/<evaluation_dir>/bfcl/<ckpt>/{results,scores}``.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from evaluate_experiment import get_checkpoints_and_revisions
from utils import get_step

BFCL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_bfcl
#SBATCH --output={log_dir}/eval_log_%x_%A_%a.out
#SBATCH --gres=gpu:{gpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_{gpu}-{service_quality}
#SBATCH --account={account}
#SBATCH --constraint={gpu}
#SBATCH --array=0-{max_index}
#SBATCH --mail-type=FAIL,TIME_LIMIT
{dependency}

set -e

module purge
module load arch/{gpu}
module load anaconda-py3/2024.06
module load cuda/12.4.1
module load gcc/14.2.0
conda activate {conda_env}

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_HUB_OFFLINE=1
# FlashInfer's sampler is not installed in the BFCL env: use vLLM's own one.
export VLLM_USE_FLASHINFER_SAMPLER=0
# bfcl spawns its vLLM server on a fixed port (1053) and only polls /health, so
# two evals scheduled on the same node would silently share one server and the
# second one gets 404 "model does not exist" on every request. Give each job
# its own port.
export LOCAL_SERVER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID * 7 + SLURM_ARRAY_TASK_ID) % 40000 ))
echo "vLLM port: $LOCAL_SERVER_PORT"

# ------------------------------
# Load checkpoint info from task list
# ------------------------------
TASK_LIST="{task_list}"
ENTRY=$(jq -r ".[$SLURM_ARRAY_TASK_ID]" "$TASK_LIST")

CKPT=$(echo "$ENTRY" | jq -r '.ckpt')
MODEL_PATH=$(echo "$ENTRY" | jq -r '.model_path')
RESULT_DIR=$(echo "$ENTRY" | jq -r '.result_dir')
SCORE_DIR=$(echo "$ENTRY" | jq -r '.score_dir')

echo "[Task $SLURM_ARRAY_TASK_ID] Running BFCL on checkpoint: $CKPT"
echo "Model path: $MODEL_PATH"
echo "Test categories: {test_category}"
echo "Result dir: $RESULT_DIR"
echo "Score dir: $SCORE_DIR"

mkdir -p "$RESULT_DIR" "$SCORE_DIR"

# The bfcl CLI resolves its dataset files (and its .env) relative to the
# berkeley-function-call-leaderboard package directory, so run from there.
BFCL_PATH="{bfcl_path}"
if [ ! -d "$BFCL_PATH" ]; then
    echo "BFCL directory not found: $BFCL_PATH (set BFCL_PATH to the gorilla/berkeley-function-call-leaderboard clone)"
    exit 1
fi
cd "$BFCL_PATH"

# 1. Generate the model responses (vLLM inference, server spawned by bfcl)
bfcl generate \\
    --model {bfcl_model} \\
    --local-model-path "$MODEL_PATH" \\
    --backend vllm \\
    --num-gpus {gpus} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --test-category {test_category} \\
    --result-dir "$RESULT_DIR"

# 2. Score the responses
bfcl evaluate \\
    --model {bfcl_model} \\
    --test-category {test_category} \\
    --result-dir "$RESULT_DIR" \\
    --score-dir "$SCORE_DIR"
"""

DEFAULT_BFCL_PATH = "$HOME/gorilla/berkeley-function-call-leaderboard"
# Single-turn AST + irrelevance, plus the multi-turn categories. multi_turn_long_context
# is left out: it is the slowest and mostly stresses context length, not the datamix.
DEFAULT_TEST_CATEGORY = (
    "non_live,live,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param"
)
# Chat template copied into checkpoints that have none (see BFCL_TEMPLATE).
DEFAULT_CHAT_TEMPLATE = (
    "$OpenLLM_OUTPUT/pretrain/luciole_serie/"
    "tokenizer_128k-arab-regional_v2_instruct_train/chat_template.jinja"
)


def launch_evaluation(
    experiment_path,
    hf_model=None,
    evaluation_dir="evaluation",
    test_category=DEFAULT_TEST_CATEGORY,
    bfcl_model="OpenLLM-France/Luciole-FC",
    conda_env="BFCL",
    gpus=1,
    gpu_memory_utilization=0.9,
    chat_template=DEFAULT_CHAT_TEMPLATE,
    dependency=None,
    force=False,
    multiple_of=None,
    min_step=None,
    last_checkpoint_only=False,
    infer_ckpt_name=False,
    is_nemo_rl=False,
    dry_run=False,
):
    """Submit the BFCL array job and return its job id (``None`` if nothing ran)."""
    experiment_path = Path(experiment_path)
    if not dry_run:
        print(f"\n# Experiment path: {experiment_path}")
        print(f"# BFCL test categories: {test_category}")

    checkpoints, revisions, ckpt_dir = get_checkpoints_and_revisions(
        experiment_path, hf_model, infer_ckpt_name, is_nemo_rl
    )
    if last_checkpoint_only:
        checkpoints = [checkpoints[-1]]
        revisions = [revisions[-1]]

    output_dir = experiment_path / evaluation_dir / "bfcl"
    log_dir = output_dir / "slurm_logs"
    job_dir = output_dir / "slurm_scripts"
    for directory in (output_dir, log_dir, job_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = []
    steps_done = []

    for ckpt, revision in zip(checkpoints, revisions):
        if isinstance(ckpt, Path):
            ckpt = ckpt.name

        # Baseline models (a plain HF repo name) carry no step in their name.
        try:
            _, step = get_step(ckpt)
        except RuntimeError:
            step = None

        if step is not None:
            if min_step and (step + 1) < min_step:
                print(
                    f"Skipping checkpoint: {ckpt}. Step {step} is less than min_step {min_step}"
                )
                continue

            if multiple_of and multiple_of != 1:
                if (step + 1) % multiple_of > 1 or step == 0:
                    print(
                        f"Skipping checkpoint: {ckpt}. Step {step + 1} is not a multiple of {multiple_of}"
                    )
                    continue

            if ckpt.endswith("-last") and (step in steps_done):
                print(f"Skipping last checkpoint: {ckpt}")
                continue

            steps_done.append(step)

        ckpt_output_dir = output_dir / (f"{ckpt}_{revision}" if revision else ckpt)
        score_dir = ckpt_output_dir / "scores"

        # BFCL writes its scores under <score_dir>/<model handle>: a non-empty
        # score dir means this checkpoint was already evaluated.
        if score_dir.is_dir() and any(score_dir.iterdir()) and not force:
            print(f"Skipping existing BFCL scores for checkpoint: {ckpt}")
            continue

        if hf_model is not None and not hf_model.startswith("/"):
            # Hugging Face hub model: let vLLM resolve it from the HF cache.
            model_path = hf_model
        else:
            model_path = str((ckpt_dir / ckpt).resolve())

        print(f"Accepting checkpoint: {ckpt}")
        tasks.append(
            {
                "ckpt": ckpt,
                "model_path": model_path,
                "result_dir": str(ckpt_output_dir.resolve() / "results"),
                "score_dir": str(score_dir.resolve()),
            }
        )

    if not tasks:
        if not dry_run:
            print("No checkpoint to evaluate with BFCL... Skipping")
        return None

    # Write list to JSON so Slurm script can read it
    task_list_path = job_dir / "task_list.json"
    with open(task_list_path, "w") as f:
        json.dump(tasks, f, indent=2)

    if not dry_run:
        print(f"Prepared {len(tasks)} tasks for array job.")

    account = os.environ.get("SLURM_ACCOUNT_GPU", "qgz@a100")
    gpu = account.split("@")[1]
    service_quality = os.environ.get("SLURM_QOS_GPU", "t3")
    time = "2:00:00" if service_quality == "dev" else "20:00:00"

    array_script = BFCL_TEMPLATE.format(
        log_dir=log_dir,
        gpu=gpu,
        gpus=gpus,
        cpus=gpus * (24 if gpu == "h100" else 8),
        account=account,
        time=time,
        service_quality=service_quality,
        dependency=f"#SBATCH --dependency=afterany:{dependency}" if dependency else "",
        max_index=len(tasks) - 1,
        task_list=task_list_path,
        conda_env=conda_env,
        bfcl_path=os.environ.get("BFCL_PATH", DEFAULT_BFCL_PATH),
        bfcl_model=bfcl_model,
        test_category=test_category,
        gpu_memory_utilization=gpu_memory_utilization,
        chat_template=chat_template,
    )

    array_filename = job_dir / "job_array_bfcl.slurm"
    with open(array_filename, "w") as f:
        f.write(array_script)

    if dry_run:
        print("sbatch", str(array_filename), f"# ({len(tasks)} tasks)")
        return None

    print("Submitting BFCL array:", array_filename)
    result = subprocess.run(
        ["sbatch", "--parsable", str(array_filename)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    job_id = result.stdout.strip()
    print(f"Launching BFCL evaluation for {experiment_path} with job id: {job_id}")
    return job_id


def get_parser():
    parser = argparse.ArgumentParser(
        description="Submit a SLURM array job running BFCL on each model checkpoint."
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
        "--evaluation_dir",
        type=str,
        default="evaluation",
        help="Sub-folder of the experiment where results are written (in its 'bfcl' folder).",
    )
    parser.add_argument(
        "--test_category",
        type=str,
        default=DEFAULT_TEST_CATEGORY,
        help="BFCL test categories to run (passed to --test-category).",
    )
    parser.add_argument(
        "--bfcl_model",
        type=str,
        default="OpenLLM-France/Luciole-FC",
        help="Model handle registered in BFCL (passed to --model).",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default="BFCL",
        help="Conda environment where the bfcl CLI is installed (see README.md).",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus to use.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of the GPU memory given to the vLLM server spawned by bfcl.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=DEFAULT_CHAT_TEMPLATE,
        help="Chat template (jinja file) copied into checkpoints that have none.",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        help="A dependency after which it should launch the evals",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, re-run BFCL even if scores already exist for a checkpoint.",
    )
    parser.add_argument(
        "--multiple_of",
        type=int,
        default=None,
        help="Only evaluate checkpoints whose step+1 is a multiple of this number.",
    )
    parser.add_argument(
        "--min_step", type=int, default=None, help="Minimum step to evaluate."
    )
    parser.add_argument(
        "--last_checkpoint_only",
        action="store_true",
        help="If set, only evaluate the last checkpoint.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="If set, do not submit jobs."
    )
    parser.add_argument(
        "--is_nemo_rl",
        action="store_true",
        help="If set, use Nemo-RL specific settings.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    launch_evaluation(
        experiment_path=args.experiment_path,
        hf_model=args.hf_model,
        evaluation_dir=args.evaluation_dir,
        test_category=args.test_category,
        bfcl_model=args.bfcl_model,
        conda_env=args.conda_env,
        gpus=args.gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        chat_template=args.chat_template,
        dependency=args.dependency,
        force=args.force,
        multiple_of=args.multiple_of,
        min_step=args.min_step,
        last_checkpoint_only=args.last_checkpoint_only,
        dry_run=args.dry_run,
        is_nemo_rl=args.is_nemo_rl,
    )
