import os
import argparse
import subprocess
import sys
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
training_path = os.path.abspath(os.path.join(current_dir,".."))
sys.path.append(training_path)
from train.slurm_launcher import submit_job

CONFIGS_DICT = {
        "debug": [
            dict(arch="llama1b", seq_length=4096, batch_size=1024, name_prefix="b1024-s4096"),
            dict(arch="nemotron22b", batch_size=512, seq_length=8192, name_prefix="b512-s8192", context_parallelism=1, tensor_parallelism=1),
        ],
        "minimal":
            [
                dict(arch="llama1b"),
                dict(arch="llama8b"),
                dict(arch="llama3b"),
                dict(arch="nemotronh8b"),
                dict(arch="nemotron22b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
                dict(arch="llama24b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
                dict(arch="nemotron8b", tensor_parallelism=2, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
                dict(arch="nemotron4b", tensor_parallelism=1, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
                dict(arch="qwen30ba3b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
            ],
        "full":[
            
            # llama8b
            dict(arch="llama8b", fp8=True),
            dict(arch="llama8b", seq_length=4096, batch_size=1024, name_prefix="b1024-s4096", context_parallelism=1),
            dict(arch="llama8b", fp8=True, seq_length=4096, batch_size=1024, name_prefix="b1024-s4096", context_parallelism=1),

            # llama3b
            dict(arch="llama3b", fp8=True),
            dict(arch="llama3b", seq_length=4096, batch_size=1024, name_prefix="b1024-s4096"),
            dict(arch="llama3b", fp8=True, seq_length=4096, batch_size=1024, name_prefix="b1024-s4096"),

            # nemotronh8b
            dict(arch="nemotronh8b", fp8=True),

            # nemotron22b
            dict(arch="nemotron22b", batch_size=512, seq_length=8192, name_prefix="b512-s8192", context_parallelism=2),
            dict(arch="nemotron22b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
            dict(arch="nemotron22b", batch_size=512, seq_length=8192, name_prefix="b512-s8192", context_parallelism=2, fp8=True),

            # nemotronh47b
            dict(arch="nemotronh47b", fp8=True, tensor_parallelism=4, pipeline_parallelism=4),

            # qwen32b
            dict(arch="qwen32b", fp8=True, tensor_parallelism=4, pipeline_parallelism=4, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
            dict(arch="qwen32b", fp8=True, tensor_parallelism=4, pipeline_parallelism=4, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),

            # llama24b
            dict(arch="llama24b", batch_size=512, seq_length=8192, name_prefix="b512-s8192", context_parallelism=2),
            dict(arch="llama24b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
            dict(arch="llama24b", batch_size=512, seq_length=8192, name_prefix="b512-s8192", context_parallelism=2, fp8=True),

            # nemotron8b
            dict(arch="nemotron8b", tensor_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),
            dict(arch="nemotron8b", tensor_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192", fp8=True),
            dict(arch="nemotron8b", tensor_parallelism=2, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),

            # nemotron4b
            dict(arch="nemotron4b", tensor_parallelism=1, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),
            dict(arch="nemotron4b", tensor_parallelism=1, batch_size=512, seq_length=8192, name_prefix="b512-s8192", fp8=True),
            dict(arch="nemotron4b", tensor_parallelism=1, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
        ],
        "extra": [
            # mistral12b
            dict(arch="mistral12b"),
            dict(arch="mistral12b", fp8=True),

            # mixtral8x7
            dict(arch="mixtral8x7", context_parallelism=2),
            dict(arch="mixtral8x7", fp8=True, context_parallelism=2),
            
            dict(arch="llama24b", batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
            
            # nemotron8b
            dict(arch="nemotron8b", tensor_parallelism=1, context_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),
            dict(arch="nemotron8b", tensor_parallelism=1, context_parallelism=2, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
            dict(arch="nemotron8b", tensor_parallelism=1, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),
            dict(arch="nemotron8b", tensor_parallelism=1, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
            dict(arch="nemotron8b", tensor_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192", fp8=True),
            dict(arch="nemotron8b", tensor_parallelism=1, context_parallelism=2, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
            dict(arch="nemotron8b", tensor_parallelism=1, context_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192", fp8=True),

            # nemotron4b
            dict(arch="nemotron4b", tensor_parallelism=2, batch_size=512, seq_length=8192, name_prefix="b512-s8192"),
            dict(arch="nemotron4b", tensor_parallelism=2, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096"),
            dict(arch="nemotron4b", tensor_parallelism=1, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
            dict(arch="nemotron4b", tensor_parallelism=1, batch_size=1024, seq_length=4096, name_prefix="b1024-s4096", fp8=True),
        ]
    }


def get_configs(mode):
    configs = CONFIGS_DICT['debug']
    if mode in ["minimal", "full", "extra"]:
        configs = configs + CONFIGS_DICT["minimal"]
    if mode in ["full", "extra"]:
        configs = configs + CONFIGS_DICT["full"]
    if mode in ["extra"]:
        configs = configs + CONFIGS_DICT["extra"]
    return configs

def launch_jobs(base_config, mode):
    configs = get_configs(mode)
    job_list = []
    job_folder_list = []
    for config in configs:
        run_config = base_config.copy()
        run_config.update(config)
        job, folder = submit_job(**run_config)
        if job:
            job_list.append(str(job))
            job_folder_list.append(folder)
    return job_list, job_folder_list

def launch_checker(job_list, job_folder_list):
    checker_job_list = []
    for job, folder in zip(job_list, job_folder_list):
        result = subprocess.run(
            [
                "sbatch",
                f"--job-name=check_{job}_result",
                "--dependency=afterany:" + job,
                f"{current_dir}/example.slurm",
                "checker.py",
                folder,
                job,
            ], 
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = int(match.group(1))
        else:
            raise ValueError("Failed to parse job ID from sbatch output.")
        checker_job_list.append(str(job_id))
    return checker_job_list

def launch_plot(job_list, input_folder, output_folder):
    subprocess.run(
        [
            "sbatch",
            f"--job-name=plot_{output_folder}",
            "--dependency=afterok:" + ":".join(job_list) if job_list else "",
            f"{current_dir}/example.slurm",
            "plot_table.py",
            input_folder,
            f"--output_folder {output_folder}",
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--output_benchmark_folder", type=str, default=None)
    parser.add_argument("--output_plot_folder", type=str, default=None)
    parser.add_argument("--mode", default="debug", choices=["debug", "minimal", "full", "extra"])
    args = parser.parse_args()

    output_benchmark_folder = args.output_benchmark_folder or str(args.num_nodes)
    base_config = dict(
        output_dir=os.path.join(os.path.join(os.getenv("OpenLLM_OUTPUT"), "ablations", "train"), output_benchmark_folder),
        num_nodes=args.num_nodes,
        config="../datamix/mock.json",
        mode="benchmark",
        name_prefix="",
        email=None,
        email_types="ALL",
        gpus_per_node=4,
        fp8=False,
        tensor_parallelism=None,
        pipeline_parallelism=None,
        seq_length=None,
        batch_size=None,
        context_parallelism=None,
        virtual_pipeline_parallelism=None,
        seed=None,
        base_checkpoint=None,
        performance_mode=False,
    )

    job_list, job_folder_list = launch_jobs(base_config, args.mode)
    checker_job_list = launch_checker(job_list, job_folder_list)
    output_plot_folder = args.output_plot_folder or f"plots_{str(args.num_nodes)}"
    launch_plot(checker_job_list, base_config['output_dir'], output_plot_folder)