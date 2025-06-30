from utils import create_parser, parse_args, create_executor
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.filters import ExtremeTokenizerFilter


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
    # title = doc.metadata.get("title")
    # if (
    #     (title is not None)
    #     and (title != "None")
    #     and (title.lower() not in doc.text[:200].lower())
    # ):
    #     out["title"] = doc.metadata.get("title")
    return out


def open_culture_subset(doc):
    if doc.metadata["collection"] in [
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
    ]:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    #################
    # Open Culture
    #################

    pipeline = [
        HuggingFaceDatasetReader(
            "PleIAs/common_corpus"
            if args.local
            else "/lustre/fsmisc/dataset/HuggingFace/PleIAs/common_corpus",
            {"split": "train"},
            streaming=True,
        ),
        slugify_metadata,
        LambdaFilter(open_culture_subset),
        LanguageFilter(
            keep_top_pairs_threshold=1,
            languages=["en", "fr", "it", "de", "es", "ar", "pt", "nl"],
            language_threshold=0.5,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered/removed/ft176"
            ),
        ),
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional",
            max_token_per_char=0.38,
            remove_digits=True,
            mode="CHUNKS",
            min_length=1000,
            separator=". ",
            replace_span="\n\n[...]\n\n",
            removed_spans_in_metadata=False,  # FOR DEBUGGING only
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/common_corpus_filtered/removed/extreme_tokenizer"
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
            f"{DATA_PATH}/common_corpus_filtered/data",
            output_filename="${open_type}/${collection}/${rank}.jsonl.gz",
            max_file_size=int(2e9),
        ),
    ]

    main_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/common_corpus_filtered/logs",
        job_name="common_corpus_filtered",
        partition="cpu_p1",
        time="20:00:00",
    )

    main_executor.run()
