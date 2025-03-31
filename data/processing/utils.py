import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter

import os


def get_data_path(args):
    main_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data", "raw_datasets")
    if args.ablation:
        main_path += '_ablation'
    os.makedirs(main_path, exist_ok=True)
    return main_path


def create_executor(pipeline, local=False, **kwargs):
    # Executor arguments
    if local:
        tasks = 1
        time = "00:10:00"
        qos = "qos_cpu-dev"
        pipeline[0].limit = 1000
    else:
        tasks = 50
        time = "05:00:00"
        qos = "qos_cpu-t3"

    if local:
        kwargs.pop("job_name", None)
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


def add_sampler_filter(pipeline):
    pipeline.insert(1, SamplerFilter(rate=0.05, seed=42))
    return pipeline

def create_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--ablation", action="store_true", help="Process a dataset for ablation")
    parser.add_argument("--local", action="store_true", help="Use a local executor")
    return parser
