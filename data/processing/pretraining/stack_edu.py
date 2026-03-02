import os
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

AWS_ACCESS_KEY_ID = "xxx"
AWS_SECRET_ACCESS_KEY = "xxx"


def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise


def default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def write_jsonl_gz_chunks(iterator, chunk_size, file_prefix):
    chunk = []
    file_index = 0

    def write_chunk(chunk, file_index):
        filename = f"{file_prefix}_{file_index:06}.jsonl.gz"
        print(f"Writing {filename} with {len(chunk)} items")
        with gzip.open(filename, "wb") as f:
            for obj in chunk:
                f.write(orjson.dumps(obj, default=default) + b"\n")

    for i, item in enumerate(tqdm.tqdm(iterator)):
        chunk.append(item)

        if i == 100:  # Test
            write_chunk(chunk, file_index)

        if len(chunk) == chunk_size:
            write_chunk(chunk, file_index)
            chunk.clear()
            file_index += 1

    # Write any remaining items
    if chunk:
        write_chunk(chunk, file_index)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--languages",
        nargs="+",
        help="List of programming languages to process",
        default=None,
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        assert args.languages is not None

        import boto3
        import gzip
        from datasets import load_dataset
        from botocore.exceptions import ClientError
        import orjson
        import tqdm
        import pandas as pd

        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
            aws_secret_access_key=os.environ.get(
                "AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY
            ),
        )
        s3 = session.client("s3")
        bucket_name = "softwareheritage"

        # For Python
        for language in args.languages:
            assert language in [
                "Python",
                "C",
                "CSharp",
                "Cpp",
                "Go",
                "Java",
                "JavaScript",
                "Markdown",
                "PHP",
                "Ruby",
                "Rust",
                "SQL",
                "Shell",
                "Swift",
                "TypeScript",
            ]

            # For Python
            num_proc = 96
            ds = load_dataset(
                "HuggingFaceTB/stack-edu", language, split="train", num_proc=num_proc
            )
            ds = ds.map(download_contents, input_columns="blob_id", num_proc=num_proc)
            ds = ds.filter(lambda x: x["download_success"])

            write_jsonl_gz_chunks(
                ds,
                chunk_size=250000,
                file_prefix="/data-server/datasets/text/raw/code/stack-edu/" + language,
            )

    else:

        def get_language(data, rank: int = 0, world_size: int = 1):
            for doc in data:
                file_path = doc.metadata["file_path"].split("/")[-1]
                language = file_path.split("_")[0].lower()
                doc.metadata["language"] = language
                yield doc

        output_path = f"{DATA_PATH}/stack-edu"

        pipeline = [
            JsonlReader(
                output_path,
            ),
            get_language,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename="data/stack_edu/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="stack_edu",
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
            job_name="hf_stack_edu",
            tasks=1,
            skip_completed=not args.force,
        )

        hf_executor.run()
