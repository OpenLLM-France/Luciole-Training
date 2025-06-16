from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.extractors.open_web_math import OpenWebMathExtractor
from datatrove.pipeline.filters.open_web_math_filter import OpenWebMathFilter
from datatrove.pipeline.filters import LanguageFilter
import json
import os
import argparse

TOTAL_TASKS = 100
MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

### Version 2
with open(
    f"{MAIN_PATH}/data/raw_data/math_extraction/stats/fqdn_fineweb2_fra_Latn/merged.json",
    "r",
) as file:
    domain_subset = json.load(file)

################
# First extraction
################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stats")
    parser.add_argument(
        "--dump",
        type=str,
        default="CC-MAIN-2024-42",
    )
    args = parser.parse_args()

    OUTPUT_PATH = f"{MAIN_PATH}/data/raw_data/math_extraction/datasets/owm/{args.dump}"

    main_processing_executor = SlurmPipelineExecutor(
        pipeline=[
            WarcReader(
                f"/lustre/fsmisc/dataset/CommonCrawl/{args.dump}/segments/",
                glob_pattern="*/warc/*",  # we want the warc files
                default_metadata={"dump": args.dump},
                domain_subset=domain_subset,
                limit=-1,
            ),
            OpenWebMathFilter(),
            OpenWebMathExtractor(),
            LanguageFilter(
                languages="fr",
                language_threshold=0.65,
                keep_top_pairs_threshold=1,
            ),
            JsonlWriter(
                f"{OUTPUT_PATH}/data",
                max_file_size=int(2e9),  # 2GB per file
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=2,
        time="10:00:00",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        logging_dir=f"{OUTPUT_PATH}/logs",
        job_name="owm",
    )

    main_processing_executor.run()
