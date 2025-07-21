from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    FT176_LANGUAGES,
)
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


def subset_filter(doc, collection_name="culture"):
    subsets = {
        "culture": [
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
        ],
        "gov": [
            "eurlex",
            "french-open-data",
            "gatt-library",
            "marianne-europe",
            "oecd",
            "sec",
            "tedeutenders",
            "un-digital-library",
            "wto",
        ],
        "sci": [
            "french-science-pile",
            "german-science-pile",
            "open-science-pile",
            "spanish-science-pile",
        ],
    }

    if collection_name == "all":
        selected_subset = sum(subsets.values(), [])  # flatten all lists
    else:
        selected_subset = subsets.get(collection_name, [])

    return doc.metadata["collection"] in selected_subset


def post_processing(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        languages = doc.metadata["language"]
        # Compute majority language
        doc.metadata["language_maj"] = max(languages, key=languages.count)
        doc.metadata.pop("num_removed_spans")
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--collection",
        type=str,
        default="culture",
    )
    parser.add_argument(
        "--jz",
        action="store_true",
        help="Use jz version of the fineweb2 dataset",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        ParquetReader(
            "/lustre/fsmisc/dataset/HuggingFace/PleIAs/common_corpus"
            if args.jz
            else "hf://datasets/PleIAs/common_corpus",
            glob_pattern="common_corpus_*/*.parquet",
        ),
        slugify_metadata,
        LambdaFilter(partial(subset_filter, collection_name=args.collection)),
        SplitDocument(
            min_length=1000,
            max_length=2000,
            separator="\n., ",
        ),
        LanguageFilter(
            keep_top_pairs_threshold=1,
            languages=FT176_LANGUAGES,
            language_threshold=0.5,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_chunk/removed/chunk_ft176",
            ),
        ),
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
            max_token_per_char=0.35,
            remove_digits=True,
            mode="DOCUMENT",
            batch_size=10000,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_chunk/removed/chunk_extreme_tokenizer",
            ),
        ),
        PerplexityFilter(
            use_ccnet=True,
            model_dataset="",
            language_from_metadata=True,
            min_ppl=10.0,
            max_ppl=2500.0,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_chunk/removed/chunk_ppl",
            ),
        ),
        MergeDocument(),
        post_processing,
        LambdaFilter(
            lambda doc: (
                doc.metadata["final_length"] / doc.metadata["initial_length"] > 0.5
                and len(doc.text.split()) > 50
            ),
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered_chunk/removed/doc_filtered",
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
            f"{DATA_PATH}/common_corpus_filtered_chunk/data",
            output_filename="${open_type}/${collection}/${language_maj}_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/common_corpus_filtered_chunk/logs_{args.collection}",
        job_name="ccfc",
        partition="cpu_p1" if args.jz else "prepost",
        cpus_per_task=1,
        tasks=50,
        time="20:00:00",
        tasks_per_job=1,
    )

    main_executor.run()
