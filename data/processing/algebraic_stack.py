import os

from utils import *

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "algebraic_stack"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        JsonlReader(
            "hf://datasets/EleutherAI/proof-pile-2/algebraic-stack/train",
            glob_pattern="*.jsonl.zst",
        ),
        JsonlWriter(f"{output_path}/output"),
    ]

    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
