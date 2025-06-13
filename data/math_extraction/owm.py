from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.extractors.open_web_math import OpenWebMathExtractor
from datatrove.pipeline.filters.open_web_math_filter import OpenWebMathFilter
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.filters import FastTextClassifierFilter
import json
import os

DUMP_TO_PROCESS = "CC-MAIN-2024-42"
TOTAL_TASKS = 10
FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)

OUTPUT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "data/raw_data/math_extraction/dataset/owm"
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
        FastTextClassifierFilter(
            model_url=os.path.join(
                FASTTEXT_PATH,
                "Qwen3-32B_content_edu_fra_Latn/model/topic_ngram2_epoch5_lr0.1.bin",
            ),
            keep_labels=[("science", 0.1), ("mathematics", 0.05)],
            newline_replacement=" ",
            save_labels_in_metadata=True,
            filter_name="topic",
            exclusion_writer=JsonlWriter(
                f"{OUTPUT_PATH}/removed/topic_filtering",
            ),
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
    job_name="read_warc",
)

main_processing_executor.run()
