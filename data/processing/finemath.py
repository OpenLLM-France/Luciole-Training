from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    print_builder_config,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
import os

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        choices=["finemath", "infiwebmath"],
        default="finemath",
        help="Subset to load",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.name is None:
        print_builder_config("HuggingFaceTB/finemath")

    name = f"{args.name}-3plus"

    output_path = os.path.join(DATA_PATH, "finemath", name)

    pipeline = [
        ParquetReader(
            f"hf://datasets/HuggingFaceTB/finemath/{name}",
            glob_pattern="*.parquet",
        ),
        JsonlWriter(
            f"{output_path}/data",
            output_filename="score_${int_score}_rank${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=name,
    )

    main_processing_executor.run()
