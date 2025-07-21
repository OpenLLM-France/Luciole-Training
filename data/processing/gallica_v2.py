from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    FT176_LANGUAGES,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from datatrove.pipeline.filters import (
    LanguageFilter,
    LambdaFilter,
    ExtremeTokenizerFilter,
    PerplexityFilter,
)
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.split_and_merge import SplitDocument, MergeDocument
from datatrove.data import DocumentsPipeline

mapping = {
    "monographies": "PleIAs/French-PD-Books",
    "press": "PleIAs/French-PD-Newspapers",
}


def additionnal_formatting(doc, name):
    import re

    out = {}
    author = doc.metadata.get("author")
    if author and author != "None":
        out["author"] = doc.metadata.get("author")
    date = doc.metadata.get("author")
    if date:
        out["date"] = doc.metadata.get("date")
    if name == "press":
        title = doc.metadata.get("title")
        if (
            title
            and title.lower() not in re.sub(r"\s+", " ", doc.text[:200]).strip().lower()
        ):
            out["title"] = doc.metadata.get("title")
    return out


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
        "--name",
        type=str,
        default="monographies",
        choices=list(mapping.keys()),
        help="Dataset to process",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    hf_name = mapping[args.name]

    output_path = f"{DATA_PATH}/gallica_v2/{args.name}"

    pipeline = [
        ParquetReader(
            f"hf://datasets/{hf_name}",
            glob_pattern="*.parquet",
            text_key="complete_text",
        ),
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
                f"{output_path}/removed/chunk_ft176",
            ),
        ),
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
            max_token_per_char=0.38,
            normalize_digits=True,
            mode="DOCUMENT",
            batch_size=10000,
            exclusion_writer=JsonlWriter(
                f"{output_path}/removed/chunk_extreme_tokenizer",
            ),
        ),
        PerplexityFilter(
            use_ccnet=True,
            model_dataset="",
            language_from_metadata=True,
            min_ppl=10.0,
            max_ppl=2500.0,
            exclusion_writer=JsonlWriter(
                f"{output_path}/removed/chunk_ppl",
            ),
        ),
        MergeDocument(
            min_character_ratio= 0.5,
            min_words=50,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/removed/doc_filtered",
            )
        ),
        post_processing,
        PrefixFormatter(
            date_keys=[],
            additionnal_formatting=partial(additionnal_formatting, name=args.name),
            prefix_pipeline={
                "author": "Author",
                "title": "Title",
                "date": "Date",
            },
        ),
        JsonlWriter(
            f"{output_path}/data",
            output_filename="${language_maj}_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=args.name,
        cpu_per_task=2,
        mail_type="REQUEUE",
        mail_user="ogouvert@linagora.com",
        tasks_per_job=5,
    )
    main_processing_executor.run()
