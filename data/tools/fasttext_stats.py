import argparse
import os

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.stats import StatsMerger
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig

TOTAL_TASKS = 50

class FasttextStats(BaseStats):

    name = "📈 Fasttext Stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        toxic_score = doc.metadata["is_toxic"]
        ad_score = doc.metadata["is_ad"]
        edu_score = doc.metadata["edu_score_mean"]

        return {
            "fasttext_toxic": toxic_score,
            "fasttext_ad": ad_score,
            "fasttext_edu": edu_score,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fasttext Stats")
    parser.add_argument("--reader_type", type=str, default="jsonl", choices=["jsonl", "parquet"], help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate")

    args = parser.parse_args()
    reader_type = args.reader_type
    data_path = args.data_path
    output_path = args.output_path

    if reader_type == "jsonl":
        reader = JsonlReader(data_path)
    elif reader_type == "parquet":
        reader = ParquetReader(data_path)
    else:
        raise ValueError(f"Unsupported reader type: {reader_type}")

    compute = SlurmPipelineExecutor(
        pipeline=[
            reader,
            SamplerFilter(
                rate=args.sample_rate,
            ),
            FasttextStats(
                output_folder=os.path.join(output_path, f"fasttext_stats_{args.sample_rate}/output"),
                groups_to_compute=["summary", "histogram", "fqdn", "topic"],
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time = "05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        job_name=f"fasttext-stats",
        logging_dir=os.path.join(output_path, f"fasttext_stats_{args.sample_rate}/logs_compute"),
    )

    merger = SlurmPipelineExecutor(
        pipeline=[
            StatsMerger(
                input_folder=os.path.join(output_path, f"fasttext_stats_{args.sample_rate}/output"),
                output_folder=os.path.join(output_path, f"fasttext_stats_{args.sample_rate}/output"),
                remove_input=False,
                ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time = "01:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        logging_dir=os.path.join(output_path, f"fasttext_stats_{args.sample_rate}/logs_merge"),
        job_name=f"merging-fasttext-stats",
        depends=compute,
    )

    merger.run()