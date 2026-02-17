from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
)

from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

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

    if not args.push_only:
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

    else:
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/opendata/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/opendata/data_hf",
                output_filename="data/opendata/fr/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="opendata",
                    id_key=None,
                    language="fr",
                    language_key=None,
                    conversation_key=None,
                ),
                cleanup=True,
                expand_metadata=False,
                schema=HF_SCHEMA,
            ),
        ]

        hf_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/opendata/logs_hf",
            job_name="hf_opendata",
            tasks=1,
            skip_completed=not args.force,
        )

        hf_executor.run()
