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


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    name = "starcoder"

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "allenai/olmo-mix-1124",
                {"name": name, "split": "train"},
                streaming=True,
            ),
            JsonlWriter(f"{DATA_PATH}/olmo_mix/{name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/olmo_mix/{name}/logs",
            job_name=name,
        )

        main_processing_executor.run()

    else:
        pipeline = [
            JsonlReader(f"{DATA_PATH}/olmo_mix/{name}/data"),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/olmo_mix/{name}/data_hf",
                output_filename="data/starcoder_olmomix/${extension}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="starcoder_olmomix",
                    id_key=None,
                    language=None,
                    language_key="extension",
                    conversation_key=None,
                    remove_keys=["dataset"],
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
            logging_dir=f"{DATA_PATH}/olmo_mix/{name}/logs_hf",
            job_name="hf_olmo_starcoder",
            tasks=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
