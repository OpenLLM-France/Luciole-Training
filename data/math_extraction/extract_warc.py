from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
import os
from datatrove.pipeline.extractors.megamath import MegamathReformatter
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters.open_web_math_filter import OpenWebMathFilter
from datatrove.pipeline.filters import LanguageFilter
import json

DUMP_TO_PROCESS = "CC-MAIN-2024-42"
TOTAL_TASKS = 100

OUTPUT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "data/raw_data/math_extraction/dataset"
)

with open(
    os.path.join(
        os.getenv("OpenLLM_OUTPUT"),
        "data/raw_data/math_extraction/fqdn_counts_science/merged.json",
    ),
    "r",
) as file:
    data = json.load(file)
domain_subset = {domain for domain, count in data.items() if count >= 100}

# def copy_html(
#     data: DocumentsPipeline, rank: int = 0, world_size: int = 1
# ) -> DocumentsPipeline:
#     for doc in data:
#         doc.metadata['html'] = doc.text
#         yield doc

# def use_html(
#     data: DocumentsPipeline, rank: int = 0, world_size: int = 1
# ) -> DocumentsPipeline:
#     for doc in data:
#         doc.text = doc.metadata.pop('html', "")
#         yield doc

main_processing_executor = SlurmPipelineExecutor(
    pipeline=[
        WarcReader(
            f"/lustre/fsmisc/dataset/CommonCrawl/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP_TO_PROCESS},
            domain_subset=domain_subset,
            limit=-1,
        ),
        OpenWebMathFilter(),  # for now it's the best proxy we have
        MegamathReformatter(),
        Trafilatura(),
        LanguageFilter(
            languages="fr", language_threshold=0.65, keep_top_pairs_threshold=1
        ),  # "en", "es", "de", "it"
        JsonlWriter(f"{OUTPUT_PATH}/extract/data"),
    ],
    sbatch_args={"account": "qgz@cpu"},
    tasks=TOTAL_TASKS,
    cpus_per_task=1,
    time="10:00:00",
    qos="qos_cpu-t3",
    partition="cpu_p1",
    env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
    logging_dir=f"{OUTPUT_PATH}/extract/logs",
    job_name="read_warc",
)
main_processing_executor.run()
