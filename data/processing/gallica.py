import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.filters import ExtremeTokenizerFilter
from functools import partial

# Tried GopherQualityFilter with fineweb-2 config file for french
# It took 7 minutes and 47 secondes for 1k documents...
#     Stats: {total: 538, dropped: 355, dropped_gopher_below_alpha_threshold: 328, dropped_gopher_long_doc: 26, forwarded: 183, doc_len: 17142473 [min=2233, max=592330, 93674.72±96825/doc], dropped_gopher_below_avg_threshold: 1}

mapping = {
    "monographies": "PleIAs/French-PD-Books",
    "press": "PleIAs/French-PD-Newspapers",
}


def additionnal_formatting(doc, name):
    import re

    out = {}
    # ocr = doc.metadata.get("ocr")
    # if ocr:
    #     out["ocr"] = doc.metadata.get("ocr")
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
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional",
            min_token_per_char=0,
            max_token_per_char=0.4,
            filter_mode="CHUNKS",
            replace_span="\n\n[...]\n\n",
            removed_spans_in_metadata=False,  # FOR DEBUGGING only
            exclusion_writer=JsonlWriter(f"{output_path}/removed/extreme_tokenizer"),
        ),
        PrefixFormatter(
            date_keys=[],
            additionnal_formatting=partial(additionnal_formatting, name=args.name),
            prefix_pipeline={
                "author": "Author",
                "title": "Title",
                "date": "Date",
                # "ocr": "OCR score"
            },
        ),
        JsonlWriter(f"{output_path}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=args.name,
    )
    main_processing_executor.run()
