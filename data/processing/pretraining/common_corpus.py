from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    FT176_LANGUAGES,
)
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    LanguageFilter,
    LambdaFilter,
    ExtremeTokenizerFilter,
)
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def slugify_metadata(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from slugify import slugify

    for doc in data:
        # Open type
        open_type = doc.metadata.pop("open_type")
        if open_type:
            doc.metadata["open_type"] = slugify(open_type)
        # Collection
        collection = doc.metadata.pop("collection")
        if collection:
            doc.metadata["collection"] = slugify(collection)
        # Language
        language = doc.metadata.pop("language")
        if language:
            doc.metadata["language_cc"] = slugify(language)
        yield doc


def additionnal_formatting(doc):
    out = {}
    year = doc.metadata.get("date")
    if year is not None:
        out["year"] = str(int(year))
    return out


DATASETS = {
    "culture": [
        "bnl-newspapers-1841-1879",
    ],
    "government": [
        "eurlex",
        "wto",
        "oecd",
        "gatt-library",
        "tedeutenders",
    ],
}

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    ### LOAD and SPLIT CommonCorpus
    pipeline = [
        ParquetReader(
            "hf://datasets/PleIAs/common_corpus",
            glob_pattern="common_corpus_*/*.parquet",
        ),
        slugify_metadata,
        JsonlWriter(
            f"{DATA_PATH}/common_corpus/data",
            output_filename="${open_type}/${collection}/${rank}.jsonl.gz",
        ),
    ]

    load_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/common_corpus/logs",
        job_name="load_cc",
        partition="prepost",
        cpus_per_task=1,
        tasks=50,
        time="20:00:00",
    )
    # load_executor.run()

    ### FILTER CommonCorpus
    for open_type, collections in DATASETS.items():
        for collection in collections:
            if not args.push_only:
                pipeline = [
                    JsonlReader(
                        f"{DATA_PATH}/common_corpus/data/open-{open_type}/{collection}"
                    ),
                    LanguageFilter(
                        keep_top_pairs_threshold=1,
                        languages=FT176_LANGUAGES,
                        language_threshold=0.5,
                        exclusion_writer=JsonlWriter(
                            f"{DATA_PATH}/common_corpus_filtered/removed/{collection}/ft176",
                        ),
                    ),
                    ExtremeTokenizerFilter(
                        tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
                        max_token_per_char=0.38,
                        normalize_digits=False,
                        mode="CHUNKS",
                        min_length=1000,
                        max_length=2000,
                        separator=("\n", ". ", ", ", " "),
                        replace_span="\n\n[...]\n\n",
                        removed_spans_in_metadata=False,
                        exclusion_writer=JsonlWriter(
                            f"{DATA_PATH}/common_corpus_filtered/removed/{collection}/extreme_tokenizer"
                        ),
                    ),
                    LambdaFilter(
                        lambda doc: len(doc.text.split()) > 50,
                        exclusion_writer=JsonlWriter(
                            f"{DATA_PATH}/common_corpus_filtered/removed/{collection}/too_short_doc",
                        ),
                    ),
                    PIIFormatter(remove_ips=False),
                    PhoneNumberPII(["ZZ"], replacement="<PHONE_NUMBER>"),
                    PrefixFormatter(
                        infer_date_format=True,
                        additionnal_formatting=additionnal_formatting,
                        prefix_pipeline={
                            "year": "Year",
                        },
                    ),
                    JsonlWriter(
                        f"{DATA_PATH}/common_corpus_filtered/data/{open_type}/{collection}",
                        output_filename="${language}_${rank}.jsonl.gz",
                    ),
                ]
                add_sampler_filter(pipeline, args.sample_rate)

                if collection in ["us-pd-books", "english-pd", "german-pd"]:
                    tasks = 50
                else:
                    tasks = 10
                filter_executor = create_executor(
                    pipeline,
                    local=args.local,
                    debug=args.debug,
                    logging_dir=f"{DATA_PATH}/common_corpus_filtered/logs/{collection}",
                    job_name=collection,
                    partition="prepost",
                    cpus_per_task=2,
                    tasks=tasks,
                    time="20:00:00",
                )
                filter_executor.run()

            else:
                pipeline = [
                    JsonlReader(
                        f"{DATA_PATH}/common_corpus_filtered/data/{open_type}/{collection}"
                    ),
                    HuggingFaceDatasetWriter(
                        dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                        + ("-debug" if args.debug else ""),
                        private=True,
                        local_working_dir=f"{DATA_PATH}/common_corpus_filtered/data_hf/{open_type}/{collection}",
                        output_filename=f"data/common_corpus/open-{open_type}/{collection}/"
                        + "${language}/${rank}.parquet",
                        adapter=partial(
                            _custom_adapter_for_hf,
                            source=f"common_corpus/open-{open_type}/{collection}",
                            id_key=None,
                            reset_id=True,
                            language=None,
                            language_key="language",
                            conversation_key=None,
                            remove_keys=[
                                "token_counts",
                                "char_counts",
                                "token_per_chars",
                                "language_",
                                "__index_level_0__",
                            ],
                            remove_prefix=True,
                        ),
                        cleanup=True,
                        expand_metadata=False,
                        schema=HF_SCHEMA,
                    ),
                ]

                hf_executor = create_executor(
                    pipeline,
                    local=args.local,
                    debug=args.debug,
                    logging_dir=f"{DATA_PATH}/common_corpus_filtered/logs_hf/{open_type}/{collection}",
                    job_name="hf_cc",
                    tasks=5,
                    time="20:00:00",
                    skip_completed=not args.force,
                )

                hf_executor.run()
