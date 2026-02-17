import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "starcoder"
    output_path = os.path.join(DATA_PATH, dataset_name)

    if not args.push_only:
        pipeline = [
            ParquetReader(
                "hf://datasets/bigcode/starcoderdata",
                glob_pattern="**/*.parquet",
                text_key="content",
            ),
            LambdaFilter(
                lambda doc: doc.metadata["max_stars_count"] >= 2
                if "max_stars_count" in doc.metadata
                else True,
                exclusion_writer=JsonlWriter(f"{output_path}/1_low_stars_count"),
            ),
            JsonlWriter(f"{output_path}/1_high_stars_count"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{output_path}/logs",
            job_name=dataset_name,
        )

        main_processing_executor.run()

    else:

        def get_language(data, rank: int = 0, world_size: int = 1):
            for doc in data:
                language = doc.metadata["file_path"].split("/")[-2]
                doc.metadata["language"] = language
                yield doc

        pipeline = [
            JsonlReader(
                output_path,
                glob_pattern="**/1_*_stars_count/*.jsonl.gz",
            ),
            get_language,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename="data/starcoder_data/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="starcoder_data",
                    id_key=None,
                    language=None,
                    language_key="language",
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
            logging_dir=f"{output_path}/logs_hf",
            job_name="hf_starcoder",
            tasks=20,
            workers=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
