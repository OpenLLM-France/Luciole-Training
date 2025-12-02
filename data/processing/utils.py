import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
import inspect
import warnings
import os
from datasets import load_dataset_builder
import sys
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import FastTextClassifierFilter, LambdaFilter
import dataclasses
import json
import pyarrow as pa

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")
if not MAIN_PATH:
    raise RuntimeError("Environment variable 'OpenLLM_OUTPUT' is not set or is empty.")

FT176_LANGUAGES = [
    "en",
    "fr",
    "it",
    "de",
    "es",
    "ar",
    "pt",
    "nl",
    "eu",
    "ca",
    "oc",
    "br",
    "co",
    "wa",
]


def get_edu_pipeline(fasttext_path, exclusion_writer):
    def edu_score(
        data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """
        `data` is a generator of Document. You must also return a generator of Document (yield)
        You can optionally use `rank` and `world_size` for sharding
        """
        for doc in data:
            # Handle educational score if present
            edu_score = doc.metadata.pop("edu_score", None)
            if edu_score is not None:
                edu_score_mean = sum(
                    int(label.split("__label__")[-1]) * prob
                    for label, prob in edu_score.items()
                )
                doc.metadata["edu_score_mean"] = edu_score_mean
                doc.metadata["edu_score"] = int(round(edu_score_mean))
            yield doc

    pipeline = [
        FastTextClassifierFilter(
            model_url=fasttext_path,
            newline_replacement=" ",
            filter_name="edu_score",
        ),
        edu_score,
        LambdaFilter(
            lambda doc: doc.metadata["edu_score"] > 0, exclusion_writer=exclusion_writer
        ),
    ]
    return pipeline


def print_builder_config(dataset_name):
    config_names = list(load_dataset_builder(dataset_name).builder_configs)
    print(f"Choose a name in: {config_names}")
    sys.exit(0)


def filter_kwargs_for_class(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    accepted_keys = set(sig.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted_keys}


def create_executor(pipeline, local=False, debug=False, **kwargs):
    # Debug mode
    if debug:
        pipeline[0].limit = 10
        kwargs["tasks"] = 1
        # kwargs["skip_completed"] = False
    # Executor arguments
    if local:
        local_kwargs = filter_kwargs_for_class(LocalPipelineExecutor, kwargs)
        main_processing_executor = LocalPipelineExecutor(
            pipeline=pipeline, **local_kwargs
        )
    else:
        tasks = kwargs.pop("tasks", 50)
        time = kwargs.pop("time", "20:00:00")
        qos = kwargs.pop("qos", "qos_cpu-t3")
        partition = kwargs.pop("partition", "prepost")
        cpus_per_task = kwargs.pop("cpus_per_task", 1)
        env_command = kwargs.pop(
            "env_command", "source ~/OpenLLM-BPI-Training/data/set_env.sh"
        )
        sbatch_args = kwargs.pop(
            "sbatch_args", {"account": "qgz@cpu", "hint": "nomultithread"}
        )
        slurm_kwargs = filter_kwargs_for_class(SlurmPipelineExecutor, kwargs)
        main_processing_executor = SlurmPipelineExecutor(
            pipeline=pipeline,
            sbatch_args=sbatch_args,
            tasks=tasks,
            cpus_per_task=cpus_per_task,
            time=time,
            qos=qos,
            partition=partition,
            requeue_signals=None,
            requeue=False,
            env_command=env_command,
            **slurm_kwargs,
        )
    return main_processing_executor


def add_sampler_filter(pipeline, sample_rate):
    if sample_rate < 1.0:
        pipeline.insert(1, SamplerFilter(rate=sample_rate, seed=42))
    return pipeline


def create_parser():
    DATA_PATH = os.path.join(MAIN_PATH, "data/raw_data/full_datasets")

    parser = argparse.ArgumentParser(
        "", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--push_only", action="store_true", help="Only push the data on the hub"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force and ignore completed tasks"
    )
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
            base_dir, f"subsampled_data_rate_{args.sample_rate:.2f}"
        )
        print(f"data_path overwritten: {args.data_path}")

    if args.debug:
        args.data_path += "_debug"
    return args


def _custom_adapter_for_hf(
    self,
    document,
    source,
    id_key=None,
    language=None,
    language_key=None,
    conversation_key=None,
    remove_keys=[],
):
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    metadata = data.pop("metadata")
    id = metadata.pop(id_key, document.id) if id_key else document.id
    if language_key:
        language = metadata.pop(language_key, language)
    conversation = metadata.pop(conversation_key, None) if conversation_key else None
    text = document.text
    remove_keys.append("file_path")
    for key in remove_keys:
        metadata.pop(key, None)
    data = {
        "source": source,
        "id": id,
        "language": language,
        "text": text,
        "messages": conversation,
        "metadata": json.dumps(metadata),
    }
    return data


HF_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field(
            "messages",
            pa.list_(
                pa.struct(
                    [pa.field("role", pa.string()), pa.field("content", pa.string())]
                )
            ),
            nullable=True,
        ),
        pa.field("metadata", pa.string(), nullable=True),
    ]
)
