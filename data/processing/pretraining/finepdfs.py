from utils import create_parser, parse_args, create_executor
from datatrove.io import get_datafolder
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_web_pipeline


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument("--edu", action="store_true")
    parser.add_argument(
        "--jz",
        action="store_true",
        help="Use jz version of the fineweb2 dataset",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    language = args.language
    dataset_name = "finepdfs" + ("-edu" if args.edu else "")
    input_path = f"hf://datasets/HuggingFaceFW/{dataset_name}/data/{language}/train"

    shards = get_datafolder(input_path).list_files(glob_pattern="*.parquet")
    assert shards, f"No *.parquet shard found in {input_path}"
    num_shards = len(shards)
    print(f"Found {num_shards} parquet shard(s) in {input_path}")

    pipeline = [
        ParquetReader(
            input_path,
        ),
        *get_web_pipeline(
            language=language,
            output_path=f"{DATA_PATH}/{dataset_name}/{language}",
            do_dedup=True,
            do_edu=False,
            do_pii=True,
            do_decont=False,
            # apertus_rule=True,
        ),
        JsonlWriter(
            f"{DATA_PATH}/{dataset_name}/{language}/data",
            output_filename="rank${rank}.jsonl.gz",
        ),
    ]

    main_executor = create_executor(
        pipeline,
        tasks=max(50, num_shards),
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{dataset_name}/{language}/logs",
        job_name=f"fw_{language}",
        partition="prepost",
        cpus_per_task=1,
        time="20:00:00",
        skip_completed=not args.force,
        
    )
    main_executor.run()