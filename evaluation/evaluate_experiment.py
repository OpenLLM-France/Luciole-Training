import argparse
import os
import math
import shlex
import shutil
import subprocess
from pathlib import Path
from utils import get_step

MAIN_COMMAND = """VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval {command} \\
    "$MODEL_ARG" \\
    "$TASK_TO_EVALUATE" \\
    --output-dir "$OUTPUT_DIR" \\
    {extra_args}
"""

SBATCH_ARRAY_TEMPLATE = (
    """#!/bin/bash
#SBATCH --job-name=eval_{task_name}
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
module load nccl/2.27.3-1-cuda
module load git
conda activate eval-env

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}
export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_HUB_OFFLINE=1

# ------------------------------
# Load checkpoint info from task list
# ------------------------------
TASK_LIST="{task_list}"
ENTRY=$(jq -r ".[$SLURM_ARRAY_TASK_ID]" "$TASK_LIST")

CKPT_DIR=$(echo "$ENTRY" | jq -r '.ckpt_dir')
OUTPUT_DIR=$(echo "$ENTRY" | jq -r '.output_dir')
MODEL_ARG=$(echo "$ENTRY" | jq -r '.model_arg')
TASK_TO_EVALUATE=$(echo "$ENTRY" | jq -r '.task_to_evaluate')

echo "[Task $SLURM_ARRAY_TASK_ID] Running checkpoint: $CKPT_DIR"
echo "Model args: $MODEL_ARG"
echo "Task to evaluate: $TASK_TO_EVALUATE"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: {extra_args}"

mkdir -p "$OUTPUT_DIR"

cd "$CKPT_DIR"

{extra_env_vars}

"""
    + MAIN_COMMAND
)


def init_extra_args(custom_tasks, max_samples=-1):
    extra_arg = ""
    # Custom tasks
    if custom_tasks is None:
        pass
    elif custom_tasks == "multilingual":
        extra_arg += "--custom-tasks lighteval.tasks.multilingual.tasks \\\n"
    elif custom_tasks == "smollm3":
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(current_dir, "custom_benchmarks", "smollm3_evals.py")
        extra_arg += f"--custom-tasks {custom_path} \\\n"
    else:
        current_dir = os.path.dirname(__file__)
        custom_path = os.path.join(
            current_dir, "custom_benchmarks", f"{custom_tasks}.py"
        )
        # print(f"> Using custom tasks from: {custom_path}")
        extra_arg += f"--custom-tasks {custom_path} \\\n"
    # Max samples
    if max_samples > 0:
        extra_arg += f"--max-samples {max_samples} \\\n"
    return extra_arg


def get_hf_model(hf_model):
    ckpt_dir = Path(".")
    if hf_model == "OLMo-2-0425-1B":  # allenai/
        revisions = [
            f"stage1-step{i*100000}-tokens{math.ceil(i*209.72)}B" for i in range(1, 19)
        ]
        checkpoints = ["allenai/" + hf_model] * len(revisions)
    elif hf_model == "OLMo-2-1124-7B":  # allenai/
        revisions = [
            f"stage1-step{i*50000}-tokens{math.ceil(i*209.72)}B"
            for i in range(1, 18)
            if i != 2
        ]
        checkpoints = ["allenai/" + hf_model] * len(revisions)
    elif hf_model == "OLMo-2-1124-13B":  # allenai/
        revisions = [
            f"stage1-step{i*25000}-tokens{math.ceil(i*209.72)}B" for i in range(1, 19)
        ]
        checkpoints = ["allenai/" + hf_model] * len(revisions)
    elif hf_model == "OLMo-2-0325-32B":  # allenai/
        revisions = [
            f"stage1-step{i*25000}-tokens{math.ceil(i*209.72)}B"
            for i in range(1, 19)
            if i != 14
        ]
        checkpoints = ["allenai/" + hf_model] * len(revisions)
    elif hf_model == "Apertus-8B-2509":  # swiss-ai/
        revisions = (
            [f"step{i*50000}-tokens{i*210}B" for i in [1] + list(range(3, 21))]
            + [f"step{i*238000 + 1194000}-tokens{i*1000 + 5014}B" for i in range(3)]
            + [f"step{i*100000 + 1800000}-tokens{i*840 + 8072}B" for i in range(9)]
        )
        checkpoints = ["swiss-ai/" + hf_model] * len(revisions)
    elif hf_model == "Lucie-7B":  # OpenLLM-France/
        revisions = [f"step{i*50000:07d}" for i in range(1, 16)]
        revisions += [
            "step0753851",
            "extension_step0001220",
        ]
        checkpoints = ["OpenLLM-France/" + hf_model] * len(revisions)
    elif hf_model == "SmolLM2-1.7B":  # HuggingFaceTB/
        checkpoints = [
            "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints" for i in range(1, 20)
        ] + ["HuggingFaceTB/SmolLM2-1.7B"]
        revisions = [f"step-{i*250000}" for i in range(1, 20)] + ["main"]
    elif hf_model == "Gaperon-1125-24B":  # almanach/
        checkpoints = [
            "step-024000_tokens-0100B-phase1",
            "step-060000_tokens-0251B-phase1",
            "step-096000_tokens-0402B-phase1",
            "step-119000_tokens-0499B-phase1",
            "step-125000_tokens-0524B-phase2",
            "step-137000_tokens-0574B-phase2",
            "step-155000_tokens-0650B-phase2",
            "step-179000_tokens-0750B-phase2",
            "step-203000_tokens-0851B-phase2",
            "step-227000_tokens-0952B-phase2",
            "step-335000_tokens-1405B-phase2",
            "step-464000_tokens-1946B-phase4",
        ]
        revisions = [""] * len(checkpoints)
        ckpt_dir = Path(
            "/lustre/fsn1/projects/rech/dmn/udd26kf/scratch/commun/hf_final_ckpts/Gaperon-24B"
        )
    elif hf_model == "Gaperon-1125-8B":  # almanach/
        checkpoints = [
            "step-0334000_tokens-0700B-phase1",
            "step-0382000_tokens-0801B-phase1",
            "step-0510000_tokens-1069B-phase1",
            "step-0590000_tokens-1237B-phase1",
            "step-0680000_tokens-1426B-phase1",
            "step-0850000_tokens-1782B-phase1",
            "step-0859000_tokens-1803B-phase2",
            "step-1023000_tokens-2491B-phase2",
            "step-1105000_tokens-2835B-phase3",
            "step-1205000_tokens-3254B-phase4",
            "step-1206000_tokens-3258B-phase5",
            "step-1361000_tokens-3909B-phase6",
            # "step-1409000_tokens-4110B-black-pepper",
        ]
        revisions = [""] * len(checkpoints)
        ckpt_dir = Path(
            "/lustre/fsn1/projects/rech/dmn/udd26kf/scratch/commun/hf_final_ckpts/Gaperon-8B"
        )
    else:
        # print(f"Selection the main revision of {hf_model} model.")
        checkpoints = [hf_model]
        revisions = ["main"]
    return checkpoints, revisions, ckpt_dir


def get_checkpoints_and_revisions(
    experiment_path, hf_model=None, infer_ckpt_name=False, is_nemo_rl=False
):
    if hf_model is not None:
        if hf_model.startswith("/"):  # Absolute path
            hf_model = Path(hf_model)
            revisions = [""]
            checkpoints = [hf_model.name]
            hf_dir = hf_model.parent
        else:
            checkpoints, revisions, hf_dir = get_hf_model(hf_model)

    else:
        hf_dir = experiment_path / "huggingface_checkpoints"
        if infer_ckpt_name:
            # Infer name of ckpt based on nemo checkpoints instead of HF conversion (needed in auto_eval.py dependency)
            experiment_name = experiment_path.name
            if not is_nemo_rl:
                ckpt_dir = experiment_path / experiment_name / "checkpoints"
                assert ckpt_dir.is_dir(), f"Directory does not exist: {ckpt_dir}"
                checkpoints = sorted(
                    [d.name.replace("=", "_") for d in ckpt_dir.iterdir() if d.is_dir()]
                )  # [::-1]
            else:
                ckpt_dir = experiment_path / "checkpoints"
                assert ckpt_dir.is_dir(), f"Directory does not exist: {ckpt_dir}"
                checkpoints = sorted(
                    [
                        f"{experiment_name}-{d.name}"
                        for d in ckpt_dir.iterdir()
                        if d.is_dir()
                    ]
                )  # [::-1]
        else:
            ckpt_dir = hf_dir
            assert ckpt_dir.is_dir(), f"Directory does not exist: {ckpt_dir}"
            checkpoints = sorted(
                [d.name for d in ckpt_dir.iterdir() if d.is_dir()]
            )  # [::-1]
        revisions = ["" for _ in checkpoints]
    return checkpoints, revisions, hf_dir


def deprecate_dir(src, dst, dry_run=False):
    """Move the ``src`` directory to ``dst`` so a fresh run starts clean.

    Used to archive existing ``results``/``details`` when re-evaluating with
    ``--force``. The ``dst`` parent is created if needed, and a numeric suffix is
    appended when ``dst`` already exists (e.g. from a previous re-run of the same
    checkpoint). Does nothing if ``src`` is not a directory.

    In ``dry_run`` mode the move is not performed: the equivalent shell command
    (``mkdir -p ... && mv ...``) is printed instead, so it can be captured in a
    generated script and executed later. The destination is computed from the
    current filesystem state, matching what a real run would do at that moment.
    """
    src = Path(src)
    if not src.is_dir():
        return None
    dst = Path(dst)
    final_dst = dst
    i = 1
    while final_dst.exists():
        final_dst = dst.with_name(f"{dst.name}_{i}")
        i += 1
    if dry_run:
        print(
            f"mkdir -p {shlex.quote(str(final_dst.parent))} && "
            f"mv {shlex.quote(str(src))} {shlex.quote(str(final_dst))}"
        )
    else:
        final_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(final_dst))
    return final_dst


def launch_evaluation(
    experiment_path,
    task_to_evaluate,
    hf_model=None,
    custom_tasks=None,
    evaluation_dir="evaluation",
    command="vllm",
    max_samples=-1,
    max_model_length=None,
    dependency=None,
    lighteval_kwargs="",
    additional_model_args=None,
    force=False,
    debug=False,
    min_step=None,
    multiple_of=None,
    last_checkpoint_only=False,
    gpus=1,
    dry_run=False,
    infer_ckpt_name=False,
    is_nemo_rl=False,
):
    experiment_path = Path(experiment_path)
    task_to_evaluate = Path(task_to_evaluate)
    if not dry_run:
        print(f"\n# Experiment path: {experiment_path}")
        print(f"# Task to evaluate: {task_to_evaluate}")

    checkpoints, revisions, ckpt_dir = get_checkpoints_and_revisions(
        experiment_path, hf_model, infer_ckpt_name, is_nemo_rl
    )
    if last_checkpoint_only:
        checkpoints = [checkpoints[-1]]
        revisions = [revisions[-1]]

    # create output dirs
    output_dir = experiment_path / evaluation_dir / task_to_evaluate.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / "slurm_scripts"
    job_dir.mkdir(parents=True, exist_ok=True)

    extra_args = init_extra_args(custom_tasks, max_samples)
    extra_args += lighteval_kwargs

    extra_env_vars = ""

    # Gaperon sets head_dim explicitly in its config, which vLLM ignores (it
    # recomputes hidden_size // num_attention_heads) -> shape mismatch at load
    # time. The sitecustomize in vllm_patches/ fixes it when the olmo2 module is
    # imported; it is a no-op for every other model, so it is set for all vLLM
    # runs rather than sniffing the checkpoint name.
    # https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1
    if command == "vllm":
        patch_dir = Path(__file__).parent.resolve() / "vllm_patches"
        extra_env_vars += f'export PYTHONPATH="{patch_dir}${{PYTHONPATH:+:$PYTHONPATH}}"\n'

    tasks = []

    steps_done = []

    for ckpt, revision in zip(checkpoints, revisions):
        if isinstance(ckpt, Path):
            ckpt = ckpt.name

        step = None

        if min_step:
            _, step = get_step(ckpt)
            if (step + 1) < min_step:
                if not dry_run:
                    print(
                        f"Skipping checkpoint: {ckpt} {revision}. Step {step} is less than min_step {min_step}"
                    )
                continue

        if multiple_of and multiple_of != 1:
            _, step = get_step(ckpt)
            if (step + 1) % multiple_of > 1:
                if not dry_run:
                    print(
                        f"Skipping checkpoint: {ckpt} {revision}. Step {step + 1} is not a multiple of {multiple_of}"
                    )
                continue

        if ckpt.endswith("-last"):  # and (multiple_of is None or step in steps_done):
            if not dry_run:
                print(f"Skipping last checkpoint: {ckpt}")
            continue

        steps_done.append(step)

        results_exist = (
            (
                (output_dir / "results" / ckpt).is_dir()
                or (output_dir / "main" / "results" / ckpt).is_dir()
                or (output_dir / "main" / "results" / os.path.split(ckpt)[-1]).is_dir()
            )
            if not revision
            else (
                ((output_dir / revision / "results").is_dir())
                or (output_dir / "results" / ckpt).is_dir()
                or (output_dir / "results" / os.path.split(ckpt)[-1]).is_dir()
            )
        )
        if results_exist and not force:
            if not dry_run:
                print(f"Skipping existing results for checkpoint: {ckpt}")
            continue

        if results_exist and force:
            # Re-evaluating with --force: move the existing results/details aside
            # (into *_deprecated) so the fresh run does not mix with the old ones.
            # In dry_run mode the moves are printed as shell commands instead of
            # being performed (kept shell-valid so the output stays executable).
            print(f"# Deprecate existing results for checkpoint: {ckpt}")
            if not revision:
                deprecate_dir(
                    output_dir / "results" / ckpt,
                    output_dir / "results_deprecated" / ckpt,
                    dry_run=dry_run,
                )
                deprecate_dir(
                    output_dir / "details" / ckpt,
                    output_dir / "details_deprecated" / ckpt,
                    dry_run=dry_run,
                )
            else:
                deprecate_dir(
                    output_dir / revision / "results",
                    output_dir / revision / "results_deprecated",
                    dry_run=dry_run,
                )
                deprecate_dir(
                    output_dir / revision / "details",
                    output_dir / revision / "details_deprecated",
                    dry_run=dry_run,
                )

        model_arg = f"model_name={ckpt},dtype=bfloat16"

        # This is needed for some models with custom code, and is not a problem for other models, so we can just set it for all
        model_arg += ",trust_remote_code=True"

        if (
            "Gaperon" in ckpt_dir.name
            and "24B" in ckpt_dir.name
            and command == "accelerate"
        ):
            model_arg += ",batch_size=1"

        if revision:
            model_arg += f",revision={revision}"

        if max_model_length:
            if command == "vllm":
                model_arg += f",max_model_length={max_model_length}"
            else:
                model_arg += f",max_length={max_model_length}"

        if additional_model_args:
            model_arg += "," + additional_model_args

        if gpus > 1:
            if command == "vllm":
                extra_env_vars += "export VLLM_HOST_IP=$(hostname -i)\nexport RAY_NODE_IP_ADDRESS=$(hostname -i)\n"
                if additional_model_args and ("parallel_size" in additional_model_args):
                    pass
                elif "ruler" in task_to_evaluate.stem.lower():
                    if gpus > 3:
                        gpu_pipeline_parallel_size = gpus // 2
                        gpu_data_parallel_size = gpus // gpu_pipeline_parallel_size
                        model_arg += f",pipeline_parallel_size={gpu_pipeline_parallel_size},data_parallel_size={gpu_data_parallel_size}"
                    else:
                        model_arg += f",pipeline_parallel_size={gpus}"
                else:
                    gpu_tensor_parallel_size = min(gpus, 4)
                    gpu_pipeline_parallel_size = gpus // gpu_tensor_parallel_size
                    model_arg += f",tensor_parallel_size={gpu_tensor_parallel_size},pipeline_parallel_size={gpu_pipeline_parallel_size}"

        # Save the tuple representing a job array element
        tasks.append(
            {
                "ckpt_dir": str(ckpt_dir.resolve()),
                "output_dir": str(
                    (output_dir if not revision else output_dir / revision).resolve()
                ),
                "model_arg": model_arg,
                "task_to_evaluate": str(task_to_evaluate.resolve()),
            }
        )

    if len(tasks) == 0:
        if not dry_run:
            print("No tasks to run... Skipping")
        return

    # Write list to JSON so Slurm script can read it
    import json

    task_list_path = job_dir / "task_list.json"
    with open(task_list_path, "w") as f:
        json.dump(tasks, f, indent=2)

    if not dry_run:
        print(f"Prepared {len(tasks)} tasks for array job.")

    account = os.environ.get("SLURM_ACCOUNT_GPU", "qgz@a100")
    gpu = account.split("@")[1]

    service_quality = os.environ.get("SLURM_QOS_GPU", "t3")
    time = "2:00:00" if service_quality == "dev" else "20:00:00"

    array_script = SBATCH_ARRAY_TEMPLATE.format(
        command=command,
        log_dir=log_dir,
        gpu=gpu,
        account=account,
        gpus=gpus,
        cpus=gpus * (24 if gpu == "h100" else 8),
        dependency=f"#SBATCH --dependency=afterany:{dependency}" if dependency else "",
        max_index=len(tasks) - 1,
        task_list=task_list_path,
        extra_args=extra_args,
        extra_env_vars=extra_env_vars,
        task_name=task_to_evaluate.stem,
        time=time,
        service_quality=service_quality,
    )

    array_filename = job_dir / "job_array_eval.slurm"
    with open(array_filename, "w") as f:
        f.write(array_script)

    if dry_run:
        print("sbatch", str(array_filename), f"# ({len(tasks)} tasks)")

        # print("cd "+tasks[0]["ckpt_dir"])
        # print(MAIN_COMMAND.format(command=command, extra_args=extra_args) \
        #       .replace("$MODEL_ARG", tasks[0]["model_arg"]) \
        #       .replace("$TASK_TO_EVALUATE", tasks[0]["task_to_evaluate"]) \
        #       .replace("$OUTPUT_DIR", tasks[0]["output_dir"]) \
        #       .replace("\\\n", "").replace("  ", " ").strip())
        # print()
    else:
        print("Submitting array:", array_filename)
        result = subprocess.run(
            ["sbatch", "--parsable", str(array_filename)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        # Example stdout: "Submitted batch job 123456"
        job_id = result.stdout.strip()
        return job_id


def get_parser():
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
        default="vllm",
        choices=["vllm", "accelerate"],
        help="",
    )
    parser.add_argument(
        "--custom_tasks",
        default=None,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=None,
        help="Forced maximum model context length.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "If set, force re-evaluation even if results exist. Existing results "
            "are moved aside (results/<ckpt> -> results_deprecated/<ckpt>, and "
            "details likewise) before the fresh run."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="If set, run in debug mode."
    )
    parser.add_argument(
        "--lighteval_kwargs",
        type=str,
        default="",
        help="Additional arguments to pass to lighteval.",
    )
    parser.add_argument(
        "--additional_model_args",
        type=str,
        default=None,
        help="Additional model args to pass to lighteval, separated by commas (e.g. 'arg1=value1,arg2=value2').",
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
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus to use.")
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
        task_to_evaluate=args.task_to_evaluate,
        hf_model=args.hf_model,
        evaluation_dir=args.evaluation_dir,
        custom_tasks=args.custom_tasks,
        command=args.command,
        max_samples=args.max_samples,
        max_model_length=args.max_model_length,
        dependency=args.dependency,
        lighteval_kwargs=args.lighteval_kwargs,
        additional_model_args=args.additional_model_args,
        force=args.force,
        debug=args.debug,
        min_step=args.min_step,
        multiple_of=args.multiple_of,
        last_checkpoint_only=args.last_checkpoint_only,
        gpus=args.gpus,
        dry_run=args.dry_run,
        is_nemo_rl=args.is_nemo_rl,
    )
