import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

import os


def get_data_path(debug=True, local=False):
    main_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "datasets")
    if debug:
        main_path += "_debug"
    elif local:
        main_path += "_local"
    os.makedirs(main_path, exist_ok=True)
    return main_path


def create_pipeline(pipeline, debug=True, local=False, **kwargs):
    # Executor arguments
    if debug or local:
        tasks = 1
        time = "00:10:00"
        qos = "qos_cpu-dev"
        pipeline[0].limit = 1000
    else:
        tasks = 50
        time = "05:00:00"
        qos = "qos_cpu-t3"
        pipeline[0].limit = -1

    if local:
        main_processing_executor = LocalPipelineExecutor(
            pipeline=pipeline, tasks=tasks, **kwargs
        )
    else:
        main_processing_executor = SlurmPipelineExecutor(
            pipeline=pipeline,
            sbatch_args={"account": "qgz@cpu"},
            tasks=tasks,
            cpus_per_task=2,
            time=time,
            qos=qos,
            partition="prepost",
            env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
            **kwargs,
        )
    return main_processing_executor


def create_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--local", action="store_true", help="Local executor")
    return parser
