import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "open_web_math"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/open-web-math/open-web-math/data",
        ),
        JsonlWriter(f"{output_path}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
