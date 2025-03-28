from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    MAIN_PATH = get_data_path(args.debug, args.local)

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/dolmino-mix-1124",
            {"name": "dclm", "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{MAIN_PATH}/dclm_dolmino/output"),
    ]

    main_processing_executor = create_pipeline(
        pipeline,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{MAIN_PATH}/dclm_dolmino/logs",
        job_name="dclm_dolmino",
    )

    main_processing_executor.run()
