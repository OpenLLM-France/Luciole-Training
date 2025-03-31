from utils import *

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "pes2o"

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/olmo-mix-1124",
            {"name": "pes2o", "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/{dataset_name}/output"),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/{dataset_name}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
