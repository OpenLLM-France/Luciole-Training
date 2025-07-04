import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

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
        output_path = os.path.join(DATA_PATH, "megamath", name)

        pipeline = [
            ParquetReader(
                f"hf://datasets/LLM360/MegaMath/{name}",
            ),
            JsonlWriter(f"{output_path}/data"),
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
