from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
import os


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


def edu_score(
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
            doc.metadata["edu_score"] = int(round(edu_score_mean))
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument(
        "--jz",
        action="store_true",
        help="Use jz version of the fineweb2 dataset",
    )
    parser.add_argument("--no_fasttext", action="store_true")
    args = parse_args(parser)
    DATA_PATH = args.data_path
    FASTTEXT_PATH = os.path.join(
        os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
    )

    dataset_name = "fineweb2_filtered"
    language = args.language

    output_dir = f"{DATA_PATH}/{dataset_name}/{language}"

    ## EDU
    if (not args.no_fasttext) and (
        language
        in [
            "fra_Latn",
            "ita_Latn",
            "spa_Latn",
            "deu_Latn",
            "nld_Latn",
            "por_Latn",
            "arb_Arab",
        ]
    ):
        fasttext_filters = [
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
            edu_score,
        ]
    else:
        fasttext_filters = []

    ## PII
    pii_cleaning = [
        PIIFormatter(
            email_replacement="<EMAIL_ADDRESS>", ip_replacement="<IP_ADDRESS>"
        ),
    ]

    if args.language == "fra_Latn":
        pii_cleaning.append(
            PhoneNumberPII(["ZZ", "FR", "CA", "BE"], replacement="<PHONE_NUMBER>")
        )
    elif args.language == "deu_Latn":
        pii_cleaning.append(PhoneNumberPII(["ZZ", "DE"], replacement="<PHONE_NUMBER>"))
    elif args.language == "spa_Latn":
        pii_cleaning.append(PhoneNumberPII(["ZZ", "ES"], replacement="<PHONE_NUMBER>"))
    elif args.language == "ita_Latn":
        pii_cleaning.append(PhoneNumberPII(["ZZ", "IT"], replacement="<PHONE_NUMBER>"))
    elif args.language == "por_Latn":
        pii_cleaning.append(PhoneNumberPII(["ZZ", "PT"], replacement="<PHONE_NUMBER>"))
    else:
        pii_cleaning = []

    # TODO
    # Decontamination

    ################
    # Pipeline
    ################

    pipeline = [
        ParquetReader(
            f"/lustre/fsmisc/dataset/HuggingFace/HuggingFaceFW/fineweb-2/data/{language}/train"
            if args.jz
            else f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
        ),
        *fasttext_filters,
        *pii_cleaning,
        AssignCluster(),
        PrefixFormatter(date_keys=["date"], date_format="%Y-%m-%dT%H:%M:%SZ"),
        JsonlWriter(
            f"{output_dir}/data",
            output_filename="edu_${edu_score}_${cluster_size_group}_rank${rank}.jsonl.gz",
            max_file_size=int(2e9),
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    executor = create_executor(
        pipeline,
        tasks=50,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_dir}/logs",
        job_name=dataset_name,
        partition="cpu_p1" if args.jz else "prepost",
    )
    executor.run()
