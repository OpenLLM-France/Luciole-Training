from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial
from datatrove.data import DocumentsPipeline

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "OpenLLM-France/wikimedia",
                {"split": "train"},
                streaming=True,
            ),
            JsonlWriter(
                f"{DATA_PATH}/wikimedia/data",
                output_filename="${language}/${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/wikimedia/logs",
            job_name="wikimedia",
            tasks=50,
        )

        main_processing_executor.run()

    else:

        def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
            for doc in data:
                doc.id = str(doc.id)
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/wikimedia/data",
            ),
            fix_data,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/wikimedia/data_hf",
                output_filename="data/wikimedia/${source}/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source=None,
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
            logging_dir=f"{DATA_PATH}/wikimedia/logs_hf",
            job_name="hf_wiki",
            tasks=5,
            skip_completed=not args.force,
        )

        hf_executor.run()
