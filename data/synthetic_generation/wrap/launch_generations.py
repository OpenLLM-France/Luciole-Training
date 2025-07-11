import os
import subprocess
import re
import time


def make_splits(filenames, max_bytes_per_split=200 * 1024 * 1024):
    """
    Splits a list of filenames into chunks where each chunk's total size does not exceed max_bytes_per_split.
    """
    current_split = []
    current_size = 0

    for filename in sorted(filenames):
        file_size = os.path.getsize(filename)
        if current_size + file_size > max_bytes_per_split:
            if current_split:
                yield current_split
            current_split = [filename]
            current_size = file_size
        else:
            current_split.append(filename)
            current_size += file_size

    if current_split:
        yield current_split

def launch_generations(
    filenames,
    expe_name,
    output_dir,
    max_num_jobs,
    *kargs, **kwargs):


    num_launched = 0

    for i, split in enumerate(make_splits(filenames)):

        jobid = launch_generation(
            split,
            f"{expe_name}_{i:02d}-{len(split)}",
            output_dir,
            *kargs, **kwargs
        )
        if jobid:
            num_launched += 1
            time.sleep(1)  # Avoid overwhelming the scheduler with too many jobs at once

        if (kwargs.get("debug") and num_launched >= 2) or (max_num_jobs and num_launched >= max_num_jobs):
            print("Debug mode: stopping after two splits.")
            return


def launch_generation(
    filenames,
    expe_name,
    output_dir,
    prompt_path,
    model_name,
    nsamples,
    max_len,
    ngpus=1,
    email=None,
    debug=False,
    ):

    if email:
        email_line = f"""#SBATCH --mail-user={email}
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL"""

    prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
    output_name = f"{model_name.split('/')[-1]}_{prompt_name}_{expe_name}"
    complete_output_dir = os.path.join(output_dir, output_name)
    if os.path.exists(complete_output_dir):
        print(f"Output directory {complete_output_dir} already exists. Please remove it or choose a different name.")
        return None
    os.makedirs(complete_output_dir, exist_ok=True)

    if debug:
        qos_lines = f"""#SBATCH --time=00:10:00 
#SBATCH --qos=qos_gpu_h100-dev"""
    else:
        qos_lines = f"""#SBATCH --time=20:00:00 
#SBATCH --qos=qos_gpu_h100-t3"""

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=generate_{output_name}
#SBATCH --output={complete_output_dir}/log.out 
#SBATCH --gres=gpu:{ngpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
{qos_lines}
#SBATCH --hint=nomultithread 
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
{email_line}

module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python generate.py {' '.join(filenames)} --expe_name {expe_name} --output_dir {output_dir} --model_name {model_name} --prompt_path {prompt_path} --max {max_len} --gpus {ngpus} --nsamples {nsamples} --disable_thinking
"""
    
    write_and_launch_slurm(slurm_content, f"{complete_output_dir}/job.slurm")
    

def write_and_launch_slurm(slurm_content, slurm_path):
    with open(slurm_path, "w") as fout:
        fout.write(slurm_content)
    print(f"Generated slurm script : {slurm_path}")
    try:
        result = subprocess.run(
            ["sbatch", slurm_path], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Job submission failed") from err
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = int(match.group(1))
    else:
        raise ValueError("Failed to parse job ID from sbatch output.")
    print(f"Job submitted {job_id}")
    return job_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch generations with SLURM.")
    parser.add_argument("filenames", nargs="+", help="List of filenames to process.")
    parser.add_argument("--expe_name", required=True, help="Experiment name.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--prompt_path", default="prompt/nemotron_cc/extract_knowledge_fr.txt", help="Path to the prompt file.")
    parser.add_argument("--model_name", default="/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-8B", help="Model name or path.")
    parser.add_argument("--nsamples", type=int, default=100000, help="Number of samples to generate.")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum length of generated text.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--email", help="Email for job notifications.", default=None)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")

    args = parser.parse_args()

    launch_generations(
        args.filenames,
        args.expe_name,
        args.output_dir,
        prompt_path=args.prompt_path,
        model_name=args.model_name,
        nsamples=args.nsamples,
        max_len=args.max_len,
        ngpus=args.ngpus,
        email=args.email,
        debug=args.debug
    )