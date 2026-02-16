from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
import os

_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import StatsMerger


class LanguageStats(BaseStats):
    name = "🔗 Language code counter"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = ["fqdn", "suffix", "summary", "histogram"],
        histogram_rounding: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        BaseStats.__init__(
            self, output_folder, groups_to_compute, histogram_rounding, top_k_config
        )

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        import re

        file_path = doc.metadata.get("file_path", None)

        # Define the regex pattern
        pattern = r"hf://datasets/bigcode/starcoderdata/([^/]+)/.*\.parquet"
        # Apply the regex
        match = re.match(pattern, file_path)
        if match:
            language = match.group(1)
        else:
            language = "not_found"
        return {
            f"language_{language}": 1,
        }


TOTAL_TASKS = 10

if __name__ == "__main__":
    subfolder = "1_high_stars_count"
    subfolder = "1_low_stars_count"

    data_path = os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), f"data/raw_datasets_ablation/starcoder/{subfolder}")
    output_path = os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), f"data/raw_datasets_ablation/starcoder/language_stats/{subfolder}")

    compute = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(data_path),
            LanguageStats(
                output_folder=os.path.join(output_path, "summary_stats/output"),
                groups_to_compute=["summary"],
            ),
        ],
        sbatch_args={"account": os.environ.get("SLURM_ACCOUNT_CPU", "qgz@cpu")},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time="05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command=f"source {_DATA_DIR}/set_env.sh",
        job_name="summary-stats",
        logging_dir=os.path.join(output_path, "summary_stats/logs_compute"),
    )

    # compute.run()

    merger = SlurmPipelineExecutor(
        pipeline=[
            StatsMerger(
                input_folder=os.path.join(output_path, "summary_stats/output"),
                output_folder=os.path.join(output_path, "summary_stats/output"),
                remove_input=False,
            ),
        ],
        sbatch_args={"account": os.environ.get("SLURM_ACCOUNT_CPU", "qgz@cpu")},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time="01:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command=f"source {_DATA_DIR}/set_env.sh",
        logging_dir=os.path.join(output_path, "summary_stats/logs_merge"),
        job_name="merging-stats",
        depends=compute,
    )

    merger.run()
