import argparse
import os

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.stats import WordStats, StatsMerger

parser = argparse.ArgumentParser(description="Summary Stats")
parser.add_argument("--dataset_name", type=str, default= "fineweb2_fra_Latn", help="")
parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate")

language_mapping = {
    "fineweb2_fra_Latn": "fr",
    "fineweb2_deu_Latn": "de",
    "fineweb2_ita_Latn": "it",
    "fineweb2_spa_Latn": "es",
    "fineweb_edu": "en",
}

main_path = os.getenv("OpenLLM_OUTPUT")
TOTAL_TASKS = 50

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    language = language_mapping.get(dataset_name, None)
    if language is None:
        raise ValueError(f"Dataset name {dataset_name} is not recognized.")
    
    folder = os.path.join(main_path, "data/data_for_tokenization/word_stats", dataset_name)

    compute = SlurmPipelineExecutor(
        pipeline=[
            ParquetReader(
                os.path.join(main_path, "data/data_for_tokenization/data", dataset_name), 
                limit=-1
                ),
            # Sampling is fine for summary stats
            SamplerFilter(
                rate=args.sample_rate,
            ),
            WordStats(
                language=language,
                output_folder=folder,
                groups_to_compute=['summary'],
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time = "05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        job_name=f"summary-stats-{dataset_name}",
        logging_dir=os.path.join(main_path, "data/data_for_tokenization/logs_word_stats", dataset_name),
    )

    # compute.run()

    merger = SlurmPipelineExecutor(
        pipeline=[
            StatsMerger(
                input_folder=folder,
                output_folder=folder,
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
        logging_dir=os.path.join(main_path, "data/data_for_tokenization/logs_word_stats_merge", dataset_name),
        job_name=f"merging-stats-{dataset_name}",
        depends=compute,
    )

    merger.run()