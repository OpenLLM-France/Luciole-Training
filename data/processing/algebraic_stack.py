import os

from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name = "algebraic_stack"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline = [
        JsonlReader(
            "hf://datasets/EleutherAI/proof-pile-2/algebraic-stack/train",
            glob_pattern="*.jsonl.zst",
        ),
        JsonlWriter(f"{output_path}/output"),
    ]

    main_processing_executor = create_pipeline(
        pipeline,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
