from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        JsonlReader(f"{DATA_PATH}/hal_cea_full_filtered"),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-BPI/Luciole-Training-Dataset"
            + ("-debug" if args.debug else ""),
            private=True,
            local_working_dir=f"{DATA_PATH}/hal_cea_full_filtered_hf/data_hf",
            output_filename="data/hal/fr/${rank}.parquet",
            adapter=partial(
                _custom_adapter_for_hf,
                source="hal",
                id_key=None,
                language="fr",
                language_key=None,
                conversation_key=None,
                remove_keys=[],
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
        logging_dir=f"{DATA_PATH}/hal_cea_full_filtered_hf/logs_hf",
        job_name="hf_hal",
        tasks=1,
        skip_completed=not args.force,
        depends=None,
    )

    hf_executor.run()
