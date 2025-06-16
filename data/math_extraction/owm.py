from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.extractors.open_web_math import OpenWebMathExtractor
from datatrove.pipeline.filters.open_web_math_filter import OpenWebMathFilter
from datatrove.pipeline.filters import LanguageFilter
import json
import os

DUMP_TO_PROCESS = "CC-MAIN-2024-42"
TOTAL_TASKS = 200
FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)

OUTPUT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "data/raw_data/math_extraction/dataset/owm_v2"
)

### Version 2
with open(
    os.path.join(
        os.getenv("OpenLLM_OUTPUT"),
        "data/raw_data/math_extraction/stats/fqdn_fineweb2_fra_Latn/merged.json",
    ),
    "r",
) as file:
    domain_subset = json.load(file)

################
# First extraction
################

main_processing_executor = SlurmPipelineExecutor(
    pipeline=[
        WarcReader(
            f"/lustre/fsmisc/dataset/CommonCrawl/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP_TO_PROCESS},
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
