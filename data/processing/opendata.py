from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
)

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

SPLITS = [
    "acco",
    "balo",
    "capp",
    "cass",
    "cnil",
    "constit",
    "debats",
    "dole",
    "inca",
    "jade",
    "jorf",
    "kali",
    "legi",
    "qr",
    "sarde",
]

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    for split in SPLITS:
        pipeline = [
            HuggingFaceDatasetReader(
                "Nicolas-BZRD/DILA_OPENDATA_FR_2023",
                {"name": "default", "split": split},
                streaming=True,
            ),
            JsonlWriter(f"{DATA_PATH}/opendata/data/{split}"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/opendata/logs/{split}",
            job_name=split,
            cpus_per_task=1,
            tasks=10,
        )

        main_processing_executor.run()
