from utils import create_parser, parse_args, create_executor, MAIN_PATH

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help="Language to process",
        choices=["fr", "en"],
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    language = args.language

    pipeline = [
        JsonlReader(
            f"{MAIN_PATH}/../datasets/raw/Claire/open/{language}",
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-BPI/Luciole-Training-Dataset"
            + ("-debug" if args.debug else ""),
            private=True,
            local_working_dir=f"{DATA_PATH}/claire_hf/{language}/data_hf",
            output_filename="data/claire/${language}/${rank}.parquet",
            adapter=partial(
                _custom_adapter_for_hf,
                source="claire",
                id_key=None,
                language=None,
                language_key="language",
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
        logging_dir=f"{DATA_PATH}/claire_hf/{language}/logs_hf",
        job_name="hf_claire",
        tasks=1,
        skip_completed=not args.force,
    )

    hf_executor.run()
