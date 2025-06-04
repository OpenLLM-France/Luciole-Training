import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
import inspect
import warnings
import os


def filter_kwargs_for_class(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    accepted_keys = set(sig.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted_keys}


def create_executor(pipeline, local=False, **kwargs):
    # Executor arguments
    if local:
        tasks = 1
        pipeline[0].limit = 1000
        local_kwargs = filter_kwargs_for_class(LocalPipelineExecutor, kwargs)
        main_processing_executor = LocalPipelineExecutor(
            pipeline=pipeline, tasks=tasks, **local_kwargs
        )
    else:
        tasks = kwargs.pop("tasks", 50)
        time = kwargs.pop("time", "05:00:00")
        qos = kwargs.pop("qos", "qos_cpu-t3")
        partition = kwargs.pop("partition", "prepost")
        slurm_kwargs = filter_kwargs_for_class(SlurmPipelineExecutor, kwargs)
        main_processing_executor = SlurmPipelineExecutor(
            pipeline=pipeline,
            sbatch_args={"account": "qgz@cpu"},
            tasks=tasks,
            cpus_per_task=2,
            time=time,
            qos=qos,
            partition=partition,
            env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
            **slurm_kwargs,
        )
    return main_processing_executor


def add_sampler_filter(pipeline, sample_rate):
    if sample_rate < 1.0:
        pipeline.insert(1, SamplerFilter(rate=sample_rate, seed=42))
    return pipeline


def create_parser():
    output_dir = os.getenv("OpenLLM_OUTPUT")
    if not output_dir:
        raise RuntimeError(
            "Environment variable 'OpenLLM_OUTPUT' is not set or is empty."
        )
    DATA_PATH = os.path.join(output_dir, "data/raw_data/full_datasets")

    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
        type=str,
        help="Where to store the process data",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Process a dataset for ablation DEPRECATED",
    )
    parser.add_argument("--local", action="store_true", help="Use a local executor")
    parser.add_argument(
        "--sample_rate",
        default=1.0,
        type=float,
        help="Sampling rate when reading the data",
    )
    return parser


def parse_args(parser):
    args = parser.parse_args()
    assert 0.0 <= args.sample_rate <= 1.0, "sample_rate must be between 0.0 and 1.0"

    if getattr(args, "ablation", False):
        args.sample_rate = 0.05
        warnings.warn(
            "--ablation is deprecated; use --sample_rate instead.", DeprecationWarning
        )
        del args.ablation  # Remove deprecated argument

    if args.sample_rate < 1.0:
        base_dir = os.path.dirname(args.data_path)
        args.data_path = os.path.join(
            base_dir, f"sample_datasets_rate_{args.sample_rate:.2f}"
        )
        print(f"data_path overwritten: {args.data_path}")
    return args
