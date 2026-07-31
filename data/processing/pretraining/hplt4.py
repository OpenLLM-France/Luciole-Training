import os

from utils import create_parser, parse_args, create_executor, MAIN_PATH

from datatrove.data import DocumentsPipeline
from datatrove.io import get_datafolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_web_pipeline

# Metadata kept as-is. Everything else is dropped: the `xml` and `md` fields
# hold the same document in two other formats (they triple the size), and
# `f`/`o`/`s`/`rs`/`de`/`c` are WARC bookkeeping.
KEEP_METADATA = [
    "url",
    "ts",  # crawl timestamp
    "crawl_id",
    "cluster_size",  # size of the near-duplicate cluster this doc represents
    "doc_scores",  # 11 per-document quality signals, summed into the bucket
    "bsc-edu",
    "finepdfs-edu",
    "fineweb2-hq",
    "jql",  # 6 LLM-judge quality scores
]
# Scalar fields of the `propella-4b` LLM annotation worth carrying over.
KEEP_PROPELLA = [
    "content_quality",
    "educational_value",
    "information_density",
    "content_safety",
    "commercial_bias",
    "content_type",
]

FASTTEXT_URL = "/data-server/models/text/fasttext/multilang/fineweb_edu_annotation"

def clean_metadata(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    """Rename `u` to `url` (expected downstream) and drop the heavy fields."""
    for doc in data:
        metadata = doc.metadata
        metadata["url"] = metadata.pop("u", None)
        metadata.pop("xml", None)
        metadata.pop("md", None)
        metadata.pop("seg_langs_openlid_v3", None)
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/lustre/fswork/dataset/.HPLT/v4/clean/fra_Latn/",
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path
    INPUT_PATH = args.input_folder
    OUTPUT_PATH = f"{DATA_PATH}/hplt4_filtered/{language}"

    shards = get_datafolder(INPUT_PATH).list_files(glob_pattern="*.jsonl.zst")
    assert shards, f"No *.jsonl.zst shard found in {INPUT_PATH}"
    print(f"Reading {len(shards)} shard(s) from {INPUT_PATH}: {shards}")

    pipeline = [
        JsonlReader(
            INPUT_PATH,
            glob_pattern="*.jsonl.zst",
            compression="zstd",
        ),
        clean_metadata,
        *get_web_pipeline(
            language,
            output_path=f"{DATA_PATH}/hplt4_filtered/{language}",
            do_dedup=True,
            do_edu=True,
            do_pii=True,
            do_decont=False,
        ),
        JsonlWriter(
            f"{OUTPUT_PATH}/data",
        ),
    ]

    main_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{OUTPUT_PATH}/logs",
        job_name=f"hplt4_{language}",
        skip_completed=not args.force,
        tasks=max(50, len(shards)),
        time="20:00:00",
        partition="cpu_p1",
        limit_debug=args.limit_debug,
    )

    main_executor.run()
