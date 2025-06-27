from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.filters import FastTextClassifierFilter, LambdaFilter
from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII
from datatrove.data import DocumentsPipeline

import os

FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)


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
        "--language", type=str, default="fr", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path
    assert language == "fr", "Must change pii for other languages..."

    ############
    ### ORIGINAL
    ############

    pipeline = [
        HuggingFaceDatasetReader(
            "uonlp/CulturaX",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(
            f"{DATA_PATH}/culturax/{language}/data",
            output_filename="${source}/${rank}.jsonl.gz",
            max_file_size=int(2e9),
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/culturax/{language}/logs",
        job_name="culturax",
        tasks=50,
    )

    ############
    ### Filtered mc4
    ############
    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/culturax/{language}/data/mC4",
        ),
        FastTextClassifierFilter(
            model_url=os.path.join(
                FASTTEXT_PATH,
                "Qwen3-32B_content_edu_fra_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin",
            ),
            newline_replacement=" ",
            filter_name="edu_score",
        ),
        edu_score,
        LambdaFilter(
            lambda doc: doc.metadata["edu_score"] > 0,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/culturax_filtered/{language}/removed/toxic",
            ),
        ),
        PIIFormatter(
            email_replacement="<EMAIL_ADDRESS>", ip_replacement="<IP_ADDRESS>"
        ),
        PhoneNumberPII(["ZZ", "FR", "CA", "BE"], replacement="<PHONE_NUMBER>"),
        PrefixFormatter(date_keys=["timestamp"], date_format="%Y/%m/%d %H:%M:%S"),
        JsonlWriter(
            f"{DATA_PATH}/culturax_filtered/{language}/data",
            output_filename="${source}/${rank}.jsonl.gz",
        ),
    ]

    add_sampler_filter(pipeline, args.sample_rate)

    filtering_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/culturax_filtered/{language}/logs",
        job_name="culturax_filtered",
        tasks=50,
        partition="cpu_p1",
        depends=main_processing_executor,
    )

    filtering_executor.run()
