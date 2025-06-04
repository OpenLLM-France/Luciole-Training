from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
import os
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.extractors.megamath import MegamathExtractor
from datatrove.pipeline.filters.open_web_math_filter import OpenWebMathFilter
from datatrove.data import Document
import json

DUMP_TO_PROCESS = "CC-MAIN-2024-42"
TOTAL_TASKS = 100
OUTPUT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "data/raw_data/math_extraction/dataset"
)

with open(
    os.path.join(
        os.getenv("OpenLLM_OUTPUT"),
        "data/raw_data/math_extraction/fqdn_counts/merged.json",
    ),
    "r",
) as file:
    data = json.load(file)
domain_subset = set(data.keys())


class DomainFilter(BaseFilter):
    name = "🇫🇷 Domain Filter"

    def __init__(self, exclusion_writer: DiskWriter = None, domain_subset: set = None):
        from tldextract import TLDExtract

        super().__init__(exclusion_writer)
        self.tld_extractor = TLDExtract()
        self.domain_subset = domain_subset

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        url = doc.metadata.get("url")
        if not url:
            return False, "missing_url"
        fqdn = self.tld_extractor.extract_str(url).fqdn
        if fqdn not in self.domain_subset:
            return False, "not_french_domain"
        return True


main_processing_executor = SlurmPipelineExecutor(
    pipeline=[
        WarcReader(
            f"/lustre/fsmisc/dataset/CommonCrawl/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP_TO_PROCESS},
            limit=-1,
        ),
        DomainFilter(domain_subset=domain_subset),
        OpenWebMathFilter(
            exclusion_writer=JsonlWriter(
                f"{OUTPUT_PATH}/extract/removed_data/openwebmath"
            )
        ),
        MegamathExtractor(),
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
