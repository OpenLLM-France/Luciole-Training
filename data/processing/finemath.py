from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    print_builder_config,
)
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_robot_filter
import os

DECONT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"),
    "data/raw_data/full_datasets/decontamination_index/data",
)

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

    ### LOAD
    pipeline = [
        ParquetReader(
            f"hf://datasets/HuggingFaceTB/finemath/{name}",
            glob_pattern="*.parquet",
        ),
        JsonlWriter(
            f"{DATA_PATH}/finemath/{name}/data",
            output_filename="score_${int_score}_rank${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    load_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/finemath/{name}/logs",
        job_name=name,
    )

    ### FILTER
    pipeline = [
        JsonlReader(f"{DATA_PATH}/finemath/{name}/data"),
        get_robot_filter(output_path=f"{DATA_PATH}/finemath_filtered/{name}"),
        JsonlWriter(f"{DATA_PATH}/finemath_filtered/{name}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/finemath_filtered/{name}/logs",
        job_name=name,
        depends=load_executor,
        partition="cpu_p1",
    )

    filter_executor.run()
