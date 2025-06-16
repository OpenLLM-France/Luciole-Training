import argparse
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.filters import FastTextClassifierFilter
import os

FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)
TOTAL_TASKS = 100


class CountDomain(PipelineStep):
    def __init__(self, output_path):
        from tldextract import TLDExtract

        super().__init__()
        self.output_path = output_path
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

        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, f"{rank:05d}.json"), "w") as f:
            json.dump(fqdn_counter, f, indent=2)


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")
    parser = argparse.ArgumentParser(description="Stats")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf://datasets/HuggingFaceFW/fineweb-2/data/fra_Latn/train",
        help="",
    )
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--language", type=str, default="fra_Latn", help="")
    parser.add_argument("--get_science", action="store_true", help="")
    parser.add_argument("--from_jsonl", action="store_true", help="")
    args = parser.parse_args()

    language = args.language
    get_science = args.get_science
    output_path = args.output_path

    if args.get_science:
        fasttext_filters = [
            FastTextClassifierFilter(
                model_url=os.path.join(
                    FASTTEXT_PATH,
                    f"Qwen3-32B_content_edu_{language}/model/topic_ngram2_epoch5_lr0.1.bin",
                ),
                keep_labels=[("science", 0.1), ("mathematics", 0.1)],
                newline_replacement=" ",
                save_labels_in_metadata=True,
                filter_name="topic",
            ),
        ]
    else:
        fasttext_filters = []

    compute = SlurmPipelineExecutor(
        pipeline=[
            ParquetReader(args.dataset_path)
            if not args.from_jsonl
            else JsonlReader(args.dataset_path),
            *fasttext_filters,
            CountDomain(output_path=f"{output_path}/data"),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        cpus_per_task=1,
        time="05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ../set_env.sh",
        job_name="get_urls",
        logging_dir=f"{output_path}/logs",
    )

    compute.run()
