from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name = "pes2o"

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/olmo-mix-1124",
            {"name": "pes2o", "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{MAIN_PATH}/{dataset_name}/output"),
    ]

    main_processing_executor = create_pipeline(
        pipeline,
        dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{MAIN_PATH}/{dataset_name}/logs",
    )

    main_processing_executor.run()
