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


def get_time_limit_and_qos(mode, num_nodes, qos=None, time=None):
    if mode in ["debug", "benchmark"]:
        default_qos = "qos_gpu_h100-dev" if num_nodes <= 8 else "qos_gpu_h100-t3"
        default_time = "01:00:00"
    elif qos and qos == "qos_gpu_h100-as":
        default_time = "100:00:00"
    elif mode.startswith("phase") or mode == "annealing":
        if num_nodes <= 8:
            default_qos = "qos_gpu_h100-dev"
            default_time = "00:30:00"
        else:
            default_qos = "qos_gpu_h100-as"
            default_time = "100:00:00"
    else:
        raise ValueError(
            f"Unkown mode {mode}, should be debug, benchmark, phase1, phase2 or annealing."
        )

    qos = qos if qos else default_qos
    if qos == "qos_gpu_h100-dev":
        default_time = "02:00:00"
    time = time if time else default_time
    return time, qos


def generate_email_line(email, mail_type="ARRAY_TASKS,BEGIN,END,FAIL"):
    email_line = ""
    if email:
        mail_type = mail_type.upper()
        if mail_type == "ALL":
            mail_type = "ARRAY_TASKS,ALL"
        email_line = f"""#SBATCH --mail-user={email}
#SBATCH --mail-type={mail_type}"""
    return email_line


def create_slurm_script(
    job_name,
    email,
    email_types,
    output_dir,
    config,
    arch,
    num_nodes,
    gpus_per_node,
    mode,
    fp8,
    tensor_parallelism,
    pipeline_parallelism,
    seq_length,
    batch_size,
    context_parallelism,
    virtual_pipeline_parallelism,
    seed,
    base_checkpoint,
    performance_mode,
    qos,
    time,
    account,
    ckpt_intervals,
):
    time, qos = get_time_limit_and_qos(mode, num_nodes, qos, time)
    account = "wuh@h100" if not account else account

    train_path = Path(__file__).resolve().parent

    logger.info(f"🚂 Train script path: {train_path}/train_model.py")

    args = f"{config} --arch {arch} --name {job_name} --mode {mode} --output_dir {output_dir}"
    if fp8:
        args += " --fp8"
    if performance_mode:
        args += " --performance_mode"
    if tensor_parallelism:
        args += f" --tensor_parallelism {tensor_parallelism}"
    if pipeline_parallelism:
        args += f" --pipeline_parallelism {pipeline_parallelism}"
    if seq_length:
        args += f" --seq_length {seq_length}"
    if batch_size:
        args += f" --batch_size {batch_size}"
    if context_parallelism:
        args += f" --context_parallelism {context_parallelism}"
    if virtual_pipeline_parallelism:
        args += f" --virtual_pipeline_parallelism {virtual_pipeline_parallelism}"
    if seed:
        args += f" --seed {seed}"
    if base_checkpoint:
        args += f" --base_checkpoint {base_checkpoint}"
    job_log_file = "job_%j"
    if mode in ["phase1", "phase2", "annealing"]:
        job_log_file = f"job_{mode}_%j"
    if ckpt_intervals:
        args += f' --ckpt_intervals "{ckpt_intervals}"'
    args += f" --max_time_per_run {time}"

    # Contenu du script SLURM
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time={time}
#SBATCH --output={output_dir}/{job_log_file}/log.out 
#SBATCH --error={output_dir}/{job_log_file}/failed.out 
#SBATCH --hint=nomultithread 
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --constraint=h100
{generate_email_line(email, email_types)}

echo "Job name: {job_name}"
echo "Qos: {qos}"
echo "Time limit: {time}"
echo "Mode: {mode}"
echo "Nodes: {num_nodes}"
echo "Output dir: {output_dir}"

cwd=$(pwd)

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
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
module load arch/h100 nemo/2.4.0

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

echo "Arguments: {args}" 
srun torchrun $DISTRIBUTED_ARGS {train_path}/train_model.py {args}
"""
    return script


def write_launch_slurm(slurm_path, slurm_content, task="", array=None, depends=None):
    with open(slurm_path, "w") as fout:
        fout.write(slurm_content)
    logger.info(f"📝 Generated slurm script : {slurm_path}")
    command = ["sbatch"]
    if array:
        command += [f"--array=1-{array}%1"]
    if depends:
        command += [f"--dependency={depends}"]
    # command += ["--contiguous"]
    command += [slurm_path]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e}")
        logger.error(f"stedrr: {e.stderr}")
        exit(1)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = int(match.group(1))
    else:
        raise ValueError("Failed to parse job ID from sbatch output.")
    logger.info(f"✅ Job submitted ({task}) with job id : {job_id}")
    return job_id


def get_job_name(kwargs):
    job_name_parts = []
    if kwargs.get("name_prefix"):
        job_name_parts.append(kwargs["name_prefix"])
    job_name_parts.extend([kwargs["arch"], kwargs["mode"]])
    # Debug / benchmark mode
    if kwargs["mode"] in ["benchmark", "debug"]:
        config_name = os.path.splitext(os.path.basename(kwargs["config"]))[0]
        job_name_parts.append(config_name)
        job_name_parts.append(kwargs["mode"])
        if kwargs["mode"] == "benchmark":
            job_name_parts.append(f"{kwargs['num_nodes']}n")
            if kwargs.get("performance_mode"):
                job_name_parts.append("perf")
        if kwargs.get("seed"):
            job_name_parts.append(f"s{kwargs['seed']}")
        if kwargs.get("fp8"):
            job_name_parts.append("fp8")
        if kwargs.get("tensor_parallelism"):
            job_name_parts.append(f"tp{kwargs['tensor_parallelism']}")
        if kwargs.get("pipeline_parallelism"):
            job_name_parts.append(f"pp{kwargs['pipeline_parallelism']}")
        if kwargs.get("context_parallelism"):
            job_name_parts.append(f"cp{kwargs['context_parallelism']}")
        if kwargs.get("virtual_pipeline_parallelism"):
            job_name_parts.append(f"vpp{kwargs['virtual_pipeline_parallelism']}")
    elif kwargs["mode"] in ["annealing"]:
        job_name_parts.append(kwargs["mode"])
    return "_".join(job_name_parts).replace(".", "_")


def submit_job(**kwargs):
    config = kwargs["config"]
    if not os.path.exists(config):
        raise RuntimeError(f"Config : {config} does not exist")

    if kwargs["mode"] in ["phase2", "annealing"] and kwargs["base_checkpoint"] is None:
        raise ValueError(
            f"You must specify --base_checkpoints when using mode {kwargs['mode']}"
        )
    if kwargs["base_checkpoint"] is not None and kwargs["name_prefix"] is None:
        raise ValueError("You must specify --name_prefix when using --base_checkpoint")

    job_name = get_job_name(kwargs)

    xp_output_dir = os.path.join(kwargs["output_dir"], job_name)

    if kwargs["mode"] not in [
        "debug",
        "phase1",
        "phase2",
        "annealing",
    ] and os.path.exists(os.path.join(xp_output_dir, "completed.txt")):
        logger.info(
            f"⏭️ Experiment {xp_output_dir} already exists, skipping job submission. If you want to force submission, remove 'completed.txt'"
        )
        return None, xp_output_dir

    os.makedirs(xp_output_dir, exist_ok=True)
    sub_xp = ""
    array = kwargs.pop("slurm_array", None)
    depends = kwargs.pop("depends", None)

    if kwargs["mode"] in ["phase1", "phase2", "annealing"]:
        kwargs["account"] = (
            "zwy@h100" if not kwargs.get("account") else kwargs["account"]
        )
        kwargs["qos"] = "qos_gpu_h100-as" if not kwargs.get("qos") else kwargs["qos"]
        sub_xp = f"_{kwargs['mode']}"
        config = kwargs["config"]
        if config == "../datamix/mock.json":
            logger.info(
                f"⚠️ Using default datamix for {kwargs['mode']} : {config}, is it wanted ?"
            )
    args = {
        **kwargs,
        "job_name": job_name,
        "config": config,
        "output_dir": xp_output_dir,
    }
    args.pop("name_prefix")
    slurm_script = create_slurm_script(**args)

    logger.info(f"🧪 Experiment name : {job_name}")
    logger.info(f"📂 Experiment path : {xp_output_dir}")

    sbatch_script_path = os.path.join(xp_output_dir, "launch.slurm")

    config_output_dir = os.path.join(xp_output_dir, "datamix")
    os.makedirs(config_output_dir, exist_ok=True)
    shutil.copy2(config, config_output_dir)
    logger.info(f"📄 Copied datamix file : {config} to {config_output_dir}")

    job_id = write_launch_slurm(
        sbatch_script_path, slurm_script, task="train", array=array, depends=depends
    )

    sub_xp_output_dir = os.path.join(xp_output_dir, f"job{sub_xp}_{job_id}")
    os.makedirs(sub_xp_output_dir, exist_ok=True)
    command = " ".join([os.path.basename(sys.executable)] + sys.argv)
    command_path = os.path.join(sub_xp_output_dir, "command.sh")
    with open(command_path, "w") as f:
        f.write(command + "\n")
    logger.info(f"📁 Run saved in : {sub_xp_output_dir}")
    shutil.copy2(sbatch_script_path, sub_xp_output_dir)
    shutil.copy2(config, sub_xp_output_dir)

    return job_id, xp_output_dir


def create_parser():
    from utils import SUPPORTED_ARCHITECTURES

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../datamix/mock.json",
        help="Path to the datamix, should be a json or yaml file.",
        type=str,
    )
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=SUPPORTED_ARCHITECTURES,
    )
    parser.add_argument(
        "--name_prefix",
        default="",
        type=str,
        help="Prefix to add to the experiment name.",
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
        "--output_dir",
        default="",
        help="Subdirectory in --output_path where to save the experiment.",
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "ablations", "train"),
        help="Base directory where to save experiments.",
        type=str,
    )
    parser.add_argument(
        "--num_nodes", default=1, type=int, help="Number of nodes to use."
    )
    parser.add_argument(
        "--gpus_per_node", default=4, type=int, help="Number of GPUs per node to use."
    )
    parser.add_argument(
        "--mode",
        default="debug",
        type=str,
        help="Training mode, can be : debug, benchmark, phase1, phase2, annealing.",
    )
    parser.add_argument(
        "--fp8",
        default=False,
        action="store_true",
        help="If given, activates fp8 training if available.",
    )
    parser.add_argument("--tensor_parallelism", "--tp", default=None, type=int)
    parser.add_argument("--pipeline_parallelism", "--pp", default=None, type=int)
    parser.add_argument("--context_parallelism", "--cp", default=None, type=int)
    parser.add_argument(
        "--virtual_pipeline_parallelism",
        "--vpp",
        default=None,
        type=int,
        help="If -1, deactivates virtual pipeline parallelism.",
    )
    parser.add_argument(
        "--seq_length",
        default=None,
        type=int,
        help="Sequence length to use for training, overrides the model default value.",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Batch size to use for training, overrides the model default value.",
    )
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        type=str,
        help="The path to a nemo compatible checkpoint to make continual learning.",
    )
    parser.add_argument(
        "--performance_mode",
        "--perf",
        default=False,
        action="store_true",
        help="If given, activates performance_mode of the recipe if available.",
    )
    parser.add_argument(
        "--qos", default=None, help="If given, it will override the default qos."
    )
    parser.add_argument(
        "--time", default=None, help="If given, it will override the default time."
    )
    parser.add_argument(
        "--account",
        default=None,
        help="If given, it will override the default account (wuh@h100).",
    )
    parser.add_argument(
        "--slurm_array",
        default=None,
        type=int,
        help="If given, it will submit the job as a slurm array job with the given number of tasks.",
    )
    parser.add_argument(
        "--depends",
        default=None,
        type=str,
    )
    parser.add_argument("--ckpt_intervals", default=None, type=str)
    return parser


def pre_submit(args):
    if args.arch == "llama":  # backward compatibility
        logger.warning(
            "llama architecture is equal to llama1b, please switch to llama1b for more clarity"
        )
        args.arch = "llama1b"

    args_dict = vars(args)
    args_dict["output_dir"] = os.path.join(args.output_path, args.output_dir)
    args_dict.pop("output_path")

    job_id, xp_output_dir = submit_job(**args_dict)
    return job_id, xp_output_dir


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    pre_submit(args)
