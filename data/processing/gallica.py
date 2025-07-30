import os
from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.filters import ExtremeTokenizerFilter, LambdaFilter
from functools import partial

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

    # Collect data and filter OCR by scores
    output_path = os.path.join(DATA_PATH, "gallica", args.name)
    pipeline = [
        ParquetReader(
            f"hf://datasets/{hf_name}",
            glob_pattern="*.parquet",
            text_key="complete_text",
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
            removed_spans_in_metadata=False,  # FOR DEBUGGING only
            exclusion_writer=JsonlWriter(f"{output_path}/removed/extreme_tokenizer"),
        ),
        LambdaFilter(
            lambda doc: len(doc.text.split()) > 50,
            exclusion_writer=JsonlWriter(
                f"{output_path}/removed/too_short_doc",
            ),
        ),
        PrefixFormatter(
            date_keys=[],
            additionnal_formatting=partial(additionnal_formatting, name=args.name),
            prefix_pipeline={
                "author": "Auteur",
                "title": "Titre",
                "date": "Date",
            },
        ),
        JsonlWriter(
            f"{output_path}/data",
            output_filename="${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=args.name,
        # depends_job_id="1602437:1602429:1602423:1602449"
    )
    main_processing_executor.run()
