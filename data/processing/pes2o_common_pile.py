from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "common-pile/peS2o_filtered",
            {"split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/pes2o_common_pile/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/pes2o_common_pile/logs",
        job_name="pes2o_common_pile",
    )

    main_processing_executor.run()
