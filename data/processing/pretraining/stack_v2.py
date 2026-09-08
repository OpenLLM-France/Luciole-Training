import os
from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

# Stack-Edu only ships an educational classifier for 15 languages, none of them
# markup. These are pulled straight from The Stack v2 instead, which means no
# quality filtering beyond vendor/generated -- hence the sampler below.
LANGUAGES = [
    "HTML",
    "CSS",
]

BUCKET_NAME = "softwareheritage"
# Threads per task. The reader shards by parquet file (28 for HTML, 4 for CSS),
# so without threads the number of files caps how much we can parallelise.
DOWNLOAD_THREADS = 32


def download_contents_step(data, rank: int = 0, world_size: int = 1):
    """Replace each document's text (the blob_id) by the file content.

    Runs after the sampler so only the files we keep are ever downloaded. The
    client is built here rather than in the parent because botocore clients are
    not fork-safe; it is shared across threads, which is safe and is what stops
    a task from being bound by the round trip to us-east-1.
    """
    import boto3
    import gzip
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice

    # The softwareheritage bucket is public and not requester-pays, so the
    # requests are unsigned: no AWS account and no credentials needed.
    s3 = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=DOWNLOAD_THREADS,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )

    def fetch(doc):
        blob_id = doc.text
        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                doc.text = fin.read().decode("utf-8", errors="ignore")
            # text_key consumed blob_id, so keep it for provenance/dedup
            doc.metadata["blob_id"] = blob_id
            return doc
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        # Batched so we never materialise a whole shard: ThreadPoolExecutor.map
        # would otherwise submit every document up front.
        while batch := list(islice(data, DOWNLOAD_THREADS * 8)):
            for doc in pool.map(fetch, batch):
                if doc is not None:
                    yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--languages",
        nargs="+",
        help="List of programming languages to process",
        default=LANGUAGES,
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Fraction of files to keep, sampled before downloading contents",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=28,
        help="Number of parallel tasks. Capped by the number of parquet files "
        "of the language (28 for HTML, 4 for CSS)",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "stack_v2"
    output_path = os.path.join(DATA_PATH, dataset_name)

    if not args.push_only:
        for language in args.languages:
            assert language in LANGUAGES

            pipeline = [
                ParquetReader(
                    "hf://datasets/bigcode/the-stack-v2",
                    glob_pattern=f"data/{language}/*.parquet",
                    text_key="blob_id",
                ),
                LambdaFilter(
                    lambda doc: not doc.metadata["is_vendor"]
                    and not doc.metadata["is_generated"]
                ),
                download_contents_step,
                JsonlWriter(
                    f"{output_path}/data",
                    output_filename=language + "_${rank}.jsonl.gz",
                ),
            ]
            add_sampler_filter(pipeline, args.sample_rate)

            main_processing_executor = create_executor(
                pipeline,
                local=args.local,
                debug=args.debug,
                limit_debug=args.limit_debug,
                logging_dir=f"{output_path}/logs_{language}",
                job_name=f"{dataset_name}_{language}",
                tasks=args.tasks,
                workers=args.tasks,
                skip_completed=not args.force,
            )

            main_processing_executor.run()

    else:

        def get_language(data, rank: int = 0, world_size: int = 1):
            for doc in data:
                file_path = doc.metadata["file_path"].split("/")[-1]
                language = file_path.split("_")[0].lower()
                doc.metadata["language"] = language
                yield doc

        pipeline = [
            JsonlReader(f"{output_path}/data"),
            get_language,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename="data/stack_v2/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="stack_v2",
                    id_key=None,
                    language=None,
                    language_key="language",
                    conversation_key=None,
                    remove_keys=[],
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
            logging_dir=f"{output_path}/logs_hf",
            job_name=f"hf_{dataset_name}",
            tasks=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
