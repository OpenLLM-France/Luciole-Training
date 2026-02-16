import argparse
import subprocess
import os
import sys
import re
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time={time}
#SBATCH --output={output_dir}/job_%j/log.out 
#SBATCH --error={output_dir}/job_%j/failed.out 
#SBATCH --hint=nomultithread 
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --constraint={gpu}
{email_line}

echo "Job name: {name}"
echo "Qos: {qos}"
echo "Time limit: {time}"
echo "Mode: {mode}"
echo "Nodes: {num_nodes}"
echo "Output dir: {output_dir}"

cwd=$(pwd)

export OpenLLM_OUTPUT=${{OpenLLM_OUTPUT:-$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output}}

export HF_HOME=${{HF_HOME:-$qgz_ALL_CCFRSCRATCH/.cache/huggingface}}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export CEEMS_ENABLE_PERF_EVENTS=1
# export CEEMS_ENABLE_PROFILING=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export TOKENIZERS_PARALLELISM=false

module purge
module load arch/{gpu} {nemo_version}

# exec 1> >(tee -a {output_dir}/log.out >&1)
# exec 2> >(tee -a {output_dir}/failed.out >&2)

# Set environment variables for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE={gpus_per_node}  # Adjust based on your setup

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank $SLURM_NODEID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       "

echo "Arguments: {train_cmd}" 
srun torchrun $DISTRIBUTED_ARGS {train_path}/train_model.py {train_cmd}
"""


def create_slurm_script(slurm_args, train_args):
    train_cmd = dict_to_cli(train_args)
    # print(f"    with train args : {train_args}")
    # print(f"    with slurm args : {slurm_args}")

    gpu = slurm_args["account"].split("@")[-1]

    script = SLURM_TEMPLATE.format(
        **slurm_args,
        **train_args,
        train_cmd=train_cmd,
        email_line=generate_email_line(slurm_args["email"], slurm_args["email_types"]),
        train_path=Path(__file__).resolve().parent,
        gpu=gpu,
    )
    return script


def generate_email_line(email, mail_type="ARRAY_TASKS,BEGIN,END,FAIL"):
    email_line = ""
    if email:
        mail_type = mail_type.upper()
        if mail_type == "ALL":
            mail_type = "ARRAY_TASKS,ALL"
        email_line = f"""#SBATCH --mail-user={email}
#SBATCH --mail-type={mail_type}"""
    return email_line


def dict_to_cli(args_dict):
    cli_parts = []
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:  # only include True flags
                cli_parts.append(f"\\\n    --{k}")
        elif isinstance(v, str):
            cli_parts.append(f"\\\n    --{k} '{v}'")
        elif v is not None:
            cli_parts.append(f"\\\n    --{k} {v}")
    return " ".join(cli_parts)


def write_launch_slurm(
    slurm_path, slurm_content, task="", slurm_array=None, dependency=None
):
    with open(slurm_path, "w") as fout:
        fout.write(slurm_content)
    command = ["sbatch"]
    if slurm_array:
        command += [f"--array=1-{slurm_array}%1"]
    if dependency:
        command += [f"--dependency={dependency}"]
    command += [slurm_path]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        exit(1)
    # except FileNotFoundError:
    #     return 0
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = int(match.group(1))
    else:
        raise ValueError("Failed to parse job ID from sbatch output.")
    logger.info(f"✅ Job submitted ({task}) with job id : {job_id}")
    return job_id


def get_expe_name(slurm_args, train_args):
    job_name_parts = []
    name_prefix = slurm_args.get("name_prefix")
    if name_prefix:
        job_name_parts.append(name_prefix)
    job_name_parts.extend([train_args["arch"], train_args["mode"]])
    # Debug / benchmark mode
    if train_args["mode"] in ["benchmark", "debug"]:
        datamix_name = os.path.splitext(os.path.basename(train_args["datamix"]))[0]
        job_name_parts.append(datamix_name)
        if train_args["mode"] == "benchmark":
            job_name_parts.append(f"{slurm_args['num_nodes']}n")
            if train_args.get("performance_mode"):
                job_name_parts.append("perf")
        if train_args.get("fp8"):
            job_name_parts.append("fp8")
            if train_args.get("fp8_recipe"):
                job_name_parts.append(train_args["fp8_recipe"])
        if train_args.get("tensor_parallelism"):
            job_name_parts.append(f"tp{train_args['tensor_parallelism']}")
        if train_args.get("pipeline_parallelism"):
            job_name_parts.append(f"pp{train_args['pipeline_parallelism']}")
        if train_args.get("context_parallelism"):
            job_name_parts.append(f"cp{train_args['context_parallelism']}")
        if train_args.get("virtual_pipeline_parallelism"):
            job_name_parts.append(f"vpp{train_args['virtual_pipeline_parallelism']}")
        if train_args.get("micro_batch_size"):
            job_name_parts.append(f"mbs{train_args['micro_batch_size']}")

    expe_name = "_".join(job_name_parts).replace(".", "_")
    return expe_name


def submit_job(slurm_args, train_args):
    xp_output_dir = train_args["output_dir"]
    datamix = train_args["datamix"]

    # Check compatibility
    if not os.path.exists(datamix):
        raise RuntimeError(f"Datamix : {datamix} does not exist")

    if (
        train_args["mode"] in ["phase2", "annealing"]
        and train_args["base_checkpoint"] is None
    ):
        raise ValueError(
            f"You must specify --base_checkpoints when using mode {train_args['mode']}"
        )
    if train_args["base_checkpoint"] is not None and slurm_args["name_prefix"] is None:
        raise ValueError("You must specify --name_prefix when using --base_checkpoint")

    # SLURM args
    slurm_array = slurm_args.pop("slurm_array")
    dependency = slurm_args.pop("dependency")

    slurm_script = create_slurm_script(slurm_args, train_args)
    logger.info(f"🧪 Experiment name : {expe_name}")
    logger.info(f"📂 Experiment path : {xp_output_dir}")

    sbatch_script_path = os.path.join(xp_output_dir, "launch.slurm")

    job_id = write_launch_slurm(
        sbatch_script_path,
        slurm_script,
        task="train",
        slurm_array=slurm_array,
        dependency=dependency,
    )

    sub_xp_output_dir = os.path.join(xp_output_dir, f"job_{job_id}")
    os.makedirs(sub_xp_output_dir, exist_ok=True)
    command = " ".join([os.path.basename(sys.executable)] + sys.argv)
    command_path = os.path.join(sub_xp_output_dir, "command.sh")
    with open(command_path, "w") as f:
        f.write(command + "\n")
    logger.info(f"📁 Run saved in : {sub_xp_output_dir}")
    shutil.copy2(sbatch_script_path, sub_xp_output_dir)
    shutil.copy2(datamix, sub_xp_output_dir)

    return job_id, xp_output_dir


def get_slurm_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--name_prefix",
        default="",
        type=str,
        help="Prefix to add to the experiment name.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "test_run"),
    )
    parser.add_argument(
        "--email", default=None, type=str, help="Email to send notifications to."
    )
    parser.add_argument(
        "--email_types",
        default="ALL",
        help="Triggers used for emails (BEGIN, END, FAIL...)",
    )
    parser.add_argument(
        "--num_nodes", default=1, type=int, help="Number of nodes to use."
    )
    parser.add_argument(
        "--gpus_per_node", default=4, type=int, help="Number of GPUs per node to use."
    )
    parser.add_argument(
        "--qos",
        default="qos_gpu_h100-as",
        # choices=["qos_gpu_h100-as", "qos_gpu_h100-t3", "qos_gpu_h100-dev"],
        help="If given, it will override the default qos (qos_gpu_h100-as, qos_gpu_h100-t3, qos_gpu_h100-dev, ...).",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("SLURM_ACCOUNT_GPU", "zwy@h100"),
        help="SLURM account to use (default from $SLURM_ACCOUNT_GPU env var).",
    )
    parser.add_argument(
        "--slurm_array",
        default=None,
        type=int,
        help="If given, it will submit the job as a slurm array job with the given number of tasks.",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--nemo_version",
        default="nemo/2.3.1",
        choices=["nemo/2.3.1", "nemo/2.4.0", "nemo/2.5.2", "nemo/2.6.0"],
        type=str,
    )
    return parser


if __name__ == "__main__":
    from train_model import get_parser as get_train_parser

    train_parser = get_train_parser()
    slurm_parser = get_slurm_parser()

    # Parse each independently
    train_args, _ = train_parser.parse_known_args()
    slurm_args, _ = slurm_parser.parse_known_args()
    slurm_args = vars(slurm_args)
    train_args = vars(train_args)

    # Generate expe name
    expe_name = get_expe_name(slurm_args, train_args)
    train_args["name"] = expe_name
    print("Warning: Overriding train_args: name to", expe_name)

    # Set experiment output_dir
    output_dir = slurm_args.pop("output_dir")
    train_args["output_dir"] = os.path.join(output_dir, expe_name)
    os.makedirs(train_args["output_dir"], exist_ok=True)
    print("Warning: Overriding train_args: output_dir to", train_args["output_dir"])

    # Setup debug mode
    if train_args["mode"] in ["benchmark", "debug"]:
        if slurm_args["num_nodes"] <= 8:
            slurm_args["qos"] = "qos_gpu_h100-dev"
        train_args["time"] = "01:00:00"

    # Print args
    print("\n>> Launching with SLURM args:")
    print(slurm_args)
    print("\n>> Launching with train args:")
    print(train_args)
    print("\n")

    args_overlap = set(slurm_args) & set(train_args)
    assert not args_overlap, f"Overlapping keys found: {args_overlap}"

    job_id, xp_output_dir = submit_job(slurm_args, train_args)
