from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII, MorePIIFormatter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
import os


class PrefixFilter(BaseFilter):
    name = "🕰️ Add Prefix"

    LANG_SETTINGS = {
        "fra_Latn": {
            "locale": "fr_FR.UTF-8",
            "prompt": lambda fqdn, date: f"Source : {fqdn}\nDate : {date}\n---\n",
        },
        "spa_Latn": {
            "locale": "es_ES.UTF-8",
            "prompt": lambda fqdn, date: f"Fuente: {fqdn}\nFecha: {date}\n---\n",
        },
        "deu_Latn": {
            "locale": "de_DE.UTF-8",
            "prompt": lambda fqdn, date: f"Quelle: {fqdn}\nDatum: {date}\n---\n",
        },
        "ita_Latn": {
            "locale": "it_IT.UTF-8",
            "prompt": lambda fqdn, date: f"Fonte: {fqdn}\nData: {date}\n---\n",
        },
    }

    def __init__(
        self,
        exclusion_writer: DiskWriter | None = None,
        language: str = "fra_Latn",
    ):
        super().__init__(exclusion_writer)
        self.language = language

    def filter(self, doc: Document) -> bool:
        import tldextract
        from datetime import datetime
        import locale

        setting = self.LANG_SETTINGS[self.language]

        fqdn = tldextract.extract(doc.metadata["url"]).fqdn
        locale.setlocale(
            locale.LC_TIME, setting["locale"]
        )  # May require system support
        dt = datetime.strptime(doc.metadata["date"], "%Y-%m-%dT%H:%M:%SZ")
        formatted_date = dt.strftime("%d %B %Y")  # e.g., "24 mai 2013"
        doc.text = setting["prompt"](fqdn, formatted_date) + doc.text.strip()
        return True


class FinewebDocumentCleaning(BaseFilter):
    name = "🧹 FineWeb Document Cleaning"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        ## Line removal
        lines = doc.text.strip().splitlines()
        kept_lines = []
        for i, line in enumerate(lines, 0):
            if len(lines) > 1:
                if i == 0 and (line in lines[1]):
                    self.stat_update("first-line-included-in-second")
                    continue
                if i == 1 and (line in lines[0]):
                    self.stat_update("second-line-included-in-first")
                    continue
            if (
                kept_lines
                and (line != "")
                and ("|" not in line)
                and (line == kept_lines[-1])
            ):
                self.stat_update("consecutive-repeated-lines")
                continue
            kept_lines.append(line)
            self.stat_update("line-kept")
        doc.text = "\n".join(kept_lines)
        return True


class AssignCluster(PipelineStep):
    @staticmethod
    def get_cluster_size_group(cluster_size):
        if cluster_size in [1, 2, 3, 4]:
            return f"cluster_size-{cluster_size}"
        elif cluster_size < 100:
            return "cluster_size-5-100"
        elif cluster_size < 1000:
            return "cluster_size-100-1000"
        else:
            return "cluster_size-1000+"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        for doc in data:
            cluster_size_group = self.get_cluster_size_group(
                doc.metadata["minhash_cluster_size"]
            )
            doc.metadata["cluster_size_group"] = cluster_size_group
            yield doc


def post_process_fasttext(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    """
    `data` is a generator of Document. You must also return a generator of Document (yield)
    You can optionally use `rank` and `world_size` for sharding
    """
    for doc in data:
        # Handle educational score if present
        edu_score = doc.metadata.pop("edu_score", None)
        if edu_score is not None:
            edu_score_mean = sum(
                int(label.split("__label__")[-1]) * prob
                for label, prob in edu_score.items()
            )
            doc.metadata["edu_score_mean"] = edu_score_mean
            doc.metadata["edu_score"] = str(int(round(edu_score_mean)))
        # Handle toxicity if present
        is_toxic = doc.metadata.pop("is_toxic", None)
        if is_toxic is not None:
            doc.metadata["is_toxic"] = is_toxic["__label__true"]
        # Handle topic if present
        topic = doc.metadata.pop("topic", None)
        if topic is not None:
            doc.metadata["topic"] = max(topic, key=topic.get).replace("__label__", "")
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks to use")
    parser.add_argument(
        "--add_prefix",
        action="store_true",
        help="Add a prefix with domain source and date",
    )
    parser.add_argument("--no_fasttext", action="store_true")
    parser.add_argument(
        "--quality_criteria",
        type=str,
        default="cluster_size",
        choices=["cluster_size", "edu_score"],
        help="",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    FASTTEXT_PATH = os.path.join(
        os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
    )

    dataset_name = "fineweb2_filtered"
    language = args.language
    add_prefix = args.add_prefix
    quality_criteria = args.quality_criteria

    # Name
    # output_name = "output"
    # if add_prefix:
    #     output_name += '_wprefix'
    output_dir = f"{DATA_PATH}/{dataset_name}/{language}"

    ################
    # FastText classifier
    ################

    if (not args.no_fasttext) and (
        language in ["fra_Latn", "ita_Latn", "spa_Latn", "deu_Latn"]
    ):
        fasttext_filters = [
            FastTextClassifierFilter(
                model_url=os.path.join(
                    FASTTEXT_PATH,
                    f"Qwen3-32B_content_edu_{language}/model/is_toxic_ngram2_epoch5_lr0.1.bin",
                ),
                keep_labels=("true", 0),
                newline_replacement=" ",
                save_labels_in_metadata=True,
                filter_name="is_toxic",
            ),
            FastTextClassifierFilter(
                model_url=os.path.join(
                    FASTTEXT_PATH,
                    f"Qwen3-32B_content_edu_{language}/model/educational_score_ngram2_epoch5_lr0.1.bin",
                ),
                keep_labels=("0", 0),
                newline_replacement=" ",
                save_labels_in_metadata=True,
                filter_name="edu_score",
            ),
            FastTextClassifierFilter(
                model_url=os.path.join(
                    FASTTEXT_PATH,
                    f"Qwen3-32B_content_edu_{language}/model/topic_ngram2_epoch5_lr0.1.bin",
                ),
                keep_labels=("history", 0),
                newline_replacement=" ",
                save_labels_in_metadata=True,
                filter_name="topic",
            ),
            post_process_fasttext,
        ]
    else:
        fasttext_filters = []

    ################
    # PII
    ################

    pii_cleaning = [
        PIIFormatter(email_replacement="<<pii_email>>", ip_replacement="<<pii_ip>>"),
    ]

    if args.language == "fra_Latn":
        pii_cleaning.append(PhoneNumberPII("FR"))
        pii_cleaning.append(PhoneNumberPII("CA"))
        pii_cleaning.append(MorePIIFormatter())
    elif args.language == "deu_Latn":
        pii_cleaning.append(PhoneNumberPII("DE"))
    elif args.language == "spa_Latn":
        pii_cleaning.append(PhoneNumberPII("ES"))
    elif args.language == "ita_Latn":
        pii_cleaning.append(PhoneNumberPII("IT"))
    elif args.language == "por_Latn":
        pii_cleaning.append(PhoneNumberPII("PT"))
    else:
        pii_cleaning = []

    ################
    # Annotation
    ################

    pipeline = [
        ParquetReader(f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train"),
        *fasttext_filters,
        AssignCluster(),
        FinewebDocumentCleaning(),
        *([PrefixFilter(language=language)] if args.add_prefix else []),
        *pii_cleaning,
        JsonlWriter(f"{output_dir}/annotated_output/data", max_file_size=int(2e9)),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    annotation_executor = create_executor(
        pipeline,
        tasks=args.tasks,
        local=args.local,
        logging_dir=f"{output_dir}/annotated_output/logs",
        job_name=dataset_name,
    )

    ################
    # Writer
    ################
    if quality_criteria == "edu_score":
        writer = JsonlWriter(
            f"{output_dir}/split_by_edu_score/data",
            output_filename="edu_${edu_score}/${rank}.jsonl.gz",
            max_file_size=int(2e9),
        )
    elif quality_criteria == "cluster_size":
        writer = JsonlWriter(
            f"{output_dir}/split_by_cluster_size/data",
            output_filename="${cluster_size_group}/${rank}.jsonl.gz",
            max_file_size=int(2e9),
        )
    else:
        raise NotImplementedError()

    ################
    # Clean and split
    ################

    pipeline = [
        JsonlReader(f"{output_dir}/annotated_output/data"),
        writer,
    ]

    split_executor = create_executor(
        pipeline,
        tasks=args.tasks,
        local=args.local,
        logging_dir=f"{output_dir}/split_by_{quality_criteria}/logs",
        job_name=dataset_name,
        depends=annotation_executor,
    )

    split_executor.run()
