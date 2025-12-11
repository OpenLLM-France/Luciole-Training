from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_robot_filter

SUBSETS = [
    "megamath-web",
    "megamath-web-pro",
    "megamath-translated-code",
    "megamath-code",
    "megamath-text-code-block",
]

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        nargs="+",
        choices=SUBSETS,
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.all:
        names = SUBSETS
    else:
        names = args.name

    for name in names:
        print(f"\nProcessing {name}...")

        # Load data
        pipeline = [
            ParquetReader(
                f"hf://datasets/LLM360/MegaMath/{name}",
            ),
            JsonlWriter(f"{DATA_PATH}/megamath/{name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        load_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/megamath/{name}/logs",
            job_name=name,
        )

        # Filtering
        pipeline = [
            JsonlReader(f"{DATA_PATH}/megamath/{name}/data"),
            get_robot_filter(output_path=f"{DATA_PATH}/megamath_filtered/{name}"),
            JsonlWriter(f"{DATA_PATH}/megamath_filtered/{name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        filter_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/megamath_filtered/{name}/logs",
            job_name=name,
            depends=load_executor,
            partition="prepost",
        )

        filter_executor.run()
