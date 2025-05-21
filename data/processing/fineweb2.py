from utils import *
from pii_utils import PhoneNumberPII, MorePIIFormatter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters import FastTextClassifierFilter


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


class Rehydrater(PipelineStep):
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
        edu_score = doc.metadata.pop('edu_score')
        doc.metadata['edu_score'] = sum(int(label.split('__label__')[-1]) * prob for label, prob in edu_score.items())
        doc.metadata['is_toxic'] = doc.metadata['is_toxic']['__label__true']
        doc.metadata['is_ad'] = doc.metadata['is_ad']['__label__true']
        topic = doc.metadata.pop('topic')
        doc.metadata['top_topic'] = max(topic, key=topic.get).replace("__label__", "")
        yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument("--run-copyrights", action="store_true", help="Run copyrights")

    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "fineweb2"
    language = args.language

    ################
    ## Collect data
    ################
    
    if language == "fra_Latn": # Available only for french right now...
        fasttext_filters = [
            FastTextClassifierFilter(
                model_url = os.path.join(os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/Qwen3-32B_content_edu_400k/model/is_toxic_ngram2_epoch5_lr0.1.bin"),
                keep_labels = ("true", 0),
                newline_replacement = " ",
                save_labels_in_metadata = True,
                filter_name = "is_toxic"
            ),
            FastTextClassifierFilter(
                model_url = os.path.join(os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/Qwen3-32B_content_edu_400k/model/is_ad_ngram2_epoch5_lr0.1.bin"),
                keep_labels = ("true", 0),
                newline_replacement = " ",
                save_labels_in_metadata = True,
                filter_name = "is_ad"
            ),
            FastTextClassifierFilter(
                model_url = os.path.join(os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/Qwen3-32B_content_edu_400k/model/topic_ngram2_epoch5_lr0.1.bin"),
                keep_labels = ("history", 0),
                newline_replacement = " ",
                save_labels_in_metadata = True,
                filter_name = "topic",
            ),
            FastTextClassifierFilter(
                model_url = os.path.join(os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/Qwen3-32B_content_edu_400k/model/educational_score_ngram2_epoch5_lr0.1.bin"),
                keep_labels = ("0", 0),
                newline_replacement = " ",
                save_labels_in_metadata = True,
                filter_name = "edu_score"
            ),
            post_process_fasttext,
        ]
    else:
        fasttext_filters = []

    pipeline = [
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train"
        ),
        *fasttext_filters,
        FinewebDocumentCleaning(),
        JsonlWriter(
            f"{DATA_PATH}/{dataset_name}/data/{language}/train",
            max_file_size = int(2e9)
        ),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/{dataset_name}/logs/{language}/train",
        job_name=dataset_name,
    )

    ################
    ## Split by clusters
    ################
    pipeline = [
        JsonlReader(f"{DATA_PATH}/{dataset_name}/data/{language}/train"),
        Rehydrater(),
        JsonlWriter(
            f"{DATA_PATH}/{dataset_name}/data/{language}/clusters",
            output_filename="${cluster_size_group}/${rank}.jsonl.gz",
            max_file_size = int(2e9)  
        ),
    ]

    split_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/{dataset_name}/logs/{language}/clusters",
        job_name=dataset_name,
        depends=main_processing_executor
    )

    ################
    ## PII Cleaning
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
    elif args.language == "eng_Latn":
        pii_cleaning.append(PhoneNumberPII("US"))
        pii_cleaning.append(PhoneNumberPII("GB"))
    elif args.language == "spa_Latn":
        pii_cleaning.append(PhoneNumberPII("ES"))
    elif args.language == "ita_Latn":
        pii_cleaning.append(PhoneNumberPII("IT"))
    elif args.language == "por_Latn":
        pii_cleaning.append(PhoneNumberPII("PT"))
    else:
        pii_cleaning = []

    pipeline = [
        JsonlReader(f"{DATA_PATH}/{dataset_name}/data/{language}/clusters"),
        *pii_cleaning,
        JsonlWriter(
            f"{DATA_PATH}/{dataset_name}/data/{language}/clean_pii",
            output_filename="${cluster_size_group}/${rank}.jsonl.gz",
            max_file_size = int(2e9)  
        ),
    ]

    pii_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/{dataset_name}/logs/{language}/clean_pii",
        job_name=dataset_name,
        depends=split_executor
    )
    pii_executor.run()

