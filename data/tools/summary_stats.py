import argparse
import os

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.stats import WordStats, LineStats, DocStats, StatsMerger

parser = argparse.ArgumentParser(description="Summary Stats")
parser.add_argument("--reader_type", type=str, default="jsonl", choices=["jsonl", "parquet"], help="")
parser.add_argument("--data_path", type=str, help="")
parser.add_argument("--output_path", type=str, help="")
parser.add_argument("--language", type=str, default= "fr", help="")
parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate")

TOTAL_TASKS = 50

if __name__ == "__main__":
    args = parser.parse_args()
    reader_type = args.reader_type
    data_path = args.data_path
    output_path = args.output_path
    language = args.language

    if reader_type == "jsonl":
        reader = JsonlReader(data_path)
    elif reader_type == "parquet":
        reader = ParquetReader(data_path)
    else:
        raise ValueError(f"Unsupported reader type: {reader_type}")

    compute = SlurmPipelineExecutor(
        pipeline=[
            reader,
            # Sampling is fine for summary stats
            SamplerFilter(
                rate=args.sample_rate,
            ),
            WordStats(
                language=language,
                output_folder=os.path.join(output_path, f"summary_stats_{args.sample_rate}/output"),
                groups_to_compute=['summary'],
            ),
            # LineStats(
            #     output_folder=os.path.join(output_path, "stats/output"),
            #     groups_to_compute=['summary'],
            # ),
            # DocStats(
            #     output_folder=os.path.join(output_path, "stats/output"),
            #     groups_to_compute=['summary'],
            # ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time = "05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        job_name=f"summary-stats",
        logging_dir=os.path.join(output_path, f"summary_stats_{args.sample_rate}/logs_compute"),
    )

    # compute.run()

    merger = SlurmPipelineExecutor(
        pipeline=[
            StatsMerger(
                input_folder=os.path.join(output_path, f"summary_stats_{args.sample_rate}/output"),
                output_folder=os.path.join(output_path, f"summary_stats_{args.sample_rate}/output"),
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
        logging_dir=os.path.join(output_path, f"summary_stats_{args.sample_rate}/logs_merge"),
        job_name=f"merging-stats",
        depends=compute,
    )

    merger.run()