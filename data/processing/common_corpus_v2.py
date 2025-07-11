from utils import create_parser, parse_args, create_executor
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    LanguageFilter,
    LambdaFilter,
    ExtremeTokenizerFilter,
    PerplexityFilter,
)
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.split_and_merge import SplitDocument, MergeDocument
from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII

LANGUAGES = [
    "en",
    "fr",
    "it",
    "de",
    "es",
    "ar",
    "pt",
    "nl",
    "eu",
    "ca",
    "oc",
    "br",
    "co",
    "wa",
]


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


def subset_filter(doc):
    culture_subset = [
        "arabic-pd",
        "bnl-newspapers-1841-1879",
        "catalan-pd",
        "dutch-pd",
        "english-pd",
        "europeana",
        "german-pd",
        "german-pd-newspapers",
        "italian-pd",
        "multilingual-pd",
        "portuguese-pd",
        "spanish-pd-books",
        "spanish-pd-newspapers",
        "us-pd-books",
    ]
    gov_subset = [
        "eurlex",
        "french-open-data",
        "gatt-library",
        "marianne-europe",
        "oecd",
        "sec",
        "tedeutenders",
        "un-digital-library",
        "wto",
    ]
    sci_subset = [
        "french-science-pile",
        "german-science-pile",
        "open-science-pile",
        "spanish-science-pile",
    ]

    if doc.metadata["collection"] in culture_subset + gov_subset + sci_subset:
        return True
    return False


def post_processing(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        doc.metadata["language_agg"] = "_".join(sorted(set(doc.metadata["language"])))
        doc.metadata.pop("num_removed_spans")
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    #################
    # Open Culture
    #################

    pipeline = [
        ParquetReader(
            "hf://datasets/PleIAs/common_corpus"
            if args.local
            else "/lustre/fsmisc/dataset/HuggingFace/PleIAs/common_corpus",
            glob_pattern="common_corpus_*/*.parquet",
        ),
        slugify_metadata,
        LambdaFilter(subset_filter),
        SplitDocument(),
        LanguageFilter(
            keep_top_pairs_threshold=1,
            languages=LANGUAGES,
            language_threshold=0.5,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_v2/removed/chunk_ft176",
            ),
        ),
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
            max_token_per_char=0.35,
            remove_digits=True,
            mode="DOCUMENT",
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_v2/removed/chunk_extreme_tokenizer",
            ),
        ),
        PerplexityFilter(
            use_ccnet=True,
            model_dataset="",
            language_from_metadata=True,
            min_ppl=10.0,
            max_ppl=2500.0,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_v2/removed/chunk_ppl",
            ),
        ),
        MergeDocument(),
        post_processing,
        PIIFormatter(remove_ips=False),
        PhoneNumberPII(["ZZ"], replacement="<PHONE_NUMBER>"),
        LambdaFilter(
            lambda doc: doc.metadata["final_length"] / doc.metadata["initial_length"]
            > 0.5,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_v2/removed/doc_too_many_chunks_removed",
            ),
        ),
        LambdaFilter(
            lambda doc: len(doc.text.split()) > 50,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_v2/removed/doc_too_short",
            ),
        ),
        PrefixFormatter(
            infer_date_format=True,
            additionnal_formatting=additionnal_formatting,
            prefix_pipeline={
                "year": "Year",
            },
        ),
        JsonlWriter(
            f"{DATA_PATH}/common_corpus_filtered_v2/data",
            output_filename="${open_type}/${collection}/${language_agg}_${rank}.jsonl.gz",
        ),
    ]

    main_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/common_corpus_filtered_v2/logs",
        job_name="common_corpus_filtered_v2",
        partition="cpu_p1",
        cpus_per_task=2,  # OOM with 1...
        tasks=50,
        time="20:00:00",
    )

    main_executor.run()
