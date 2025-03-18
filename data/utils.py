
import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
import socket

KOIOS_PATH = "/media/storage0/ogouvert/datasets/training/"
JEANZAY_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training/"

hostname = socket.gethostname()

if hostname == "koios":
    MAIN_PATH = KOIOS_PATH 
else: 
    MAIN_PATH = JEANZAY_PATH

def create_pipeline(
        pipeline, dataset_name, 
        output_path, 
        debug=True,
        local=False,
        ):
    
    # Executor arguments
    if debug or local:
        tasks=1
        time="00:10:00"
        qos="qos_cpu-dev"
        pipeline[0].limit=1000
    else:
        tasks=50
        time="05:00:00"
        qos="qos_cpu-t3"
        pipeline[0].limit=-1

    if local:
        main_processing_executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=tasks, 
            logging_dir=f"{output_path}/logs",
        )
    else:
        main_processing_executor = SlurmPipelineExecutor(
            job_name=dataset_name,
            pipeline=pipeline,
            sbatch_args={"account": "qgz@cpu"},
            tasks=tasks, 
            cpus_per_task=2,
            time=time,
            logging_dir=f"{output_path}/logs",
            qos=qos,
            partition="prepost",
            env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        )
    return main_processing_executor

def create_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--local", action="store_true", help="Local executor")
    return parser
