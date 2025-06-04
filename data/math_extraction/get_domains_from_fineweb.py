import argparse
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
import os

parser = argparse.ArgumentParser(description="Stats")
parser.add_argument("--language", type=str, default="fra_Latn", help="")

TOTAL_TASKS = 50
language = "fra_Latn"


class CountDomain(PipelineStep):
    def __init__(self, output_dir):
        from tldextract import TLDExtract

        super().__init__()
        self.output_dir = output_dir
        self.tld_extractor = TLDExtract()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        from collections import Counter
        import json
        import os

        fqdn_counter = Counter()
        for doc in data:
            with self.track_time():
                fqdn = self.tld_extractor.extract_str(doc.metadata["url"]).fqdn
                fqdn_counter[fqdn] += 1
            yield doc

        os.makedirs(os.path.join(self.output_dir, "fqdn_counts"), exist_ok=True)
        with open(
            os.path.join(self.output_dir, "fqdn_counts", f"{rank:05d}.json"), "w"
        ) as f:
            json.dump(fqdn_counter, f, indent=2)


compute = SlurmPipelineExecutor(
    pipeline=[
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
        ),
        CountDomain(
            output_dir=os.path.join(
                os.getenv("OpenLLM_OUTPUT"), "data/raw_data/math_extraction"
            ),
        ),
    ],
    sbatch_args={"account": "qgz@cpu"},
    tasks=TOTAL_TASKS,
    cpus_per_task=1,
    time="05:00:00",
    qos="qos_cpu-t3",
    partition="prepost",
    env_command="source ../set_env.sh",
    job_name="get_urls",
)

compute.run()
