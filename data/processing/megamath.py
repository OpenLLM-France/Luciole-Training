import os

from utils import *

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        default="megamath-web",
        choices=['megamath-web', 'megamath-web-pro', 'megamath-translated-code', 'megamath-code', 'megamath-text-code-block'],
        help="Name of the dataset to process.",
    )
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    name = args.name
    output_path = os.path.join(DATA_PATH, "megamath", name)

    pipeline = [
        ParquetReader(
            f"hf://datasets/LLM360/MegaMath/{name}",
        ),
        JsonlWriter(f"{output_path}/data/output"),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=name,
    )

    main_processing_executor.run()
