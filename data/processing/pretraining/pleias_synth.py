import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def process_documents(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        doc.text = doc.metadata["query"].strip() + "\n\n" + doc.text.strip()
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    DATA_PATH = args.data_path
    dataset_name = "pleias_synth"
    output_path = os.path.join(DATA_PATH, dataset_name)

    if not args.push_only:
        pipeline = [
            ParquetReader(
                "hf://datasets/PleIAs/SYNTH",
                glob_pattern="*.parquet",
                text_key="synthetic_answer",
            ),
            process_documents,
            JsonlWriter(
                os.path.join(output_path, "data"),
                output_filename="${language}_${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=os.path.join(output_path, "logs"),
            job_name=dataset_name,
            tasks=20,
        )

        main_processing_executor.run()

    else:
        languages = [
            "en",
            "fr",
            "de",
            "it",
            "es",
            "pt",
            "ar",
            "nl",
            "eu",
            "oc",
            "ca",
        ]
        for language in languages:
            pipeline = [
                JsonlReader(
                    os.path.join(output_path, "data"),
                    glob_pattern=language + "_*.jsonl.gz",
                ),
                HuggingFaceDatasetWriter(
                    dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                    + ("-debug" if args.debug else ""),
                    private=True,
                    local_working_dir=f"{output_path}/{language}/data_hf",
                    output_filename=f"data/pleias_synth/{language}/"
                    + "${rank}.parquet",
                    adapter=partial(
                        _custom_adapter_for_hf,
                        source="pleias_synth",
                        id_key=None,
                        reset_id=True,
                        language=language,
                        language_key=None,
                        conversation_key=None,
                        remove_keys=[
                            "query",
                            "query_seed_text",
                            "synthetic_reasoning",
                            "constraints",
                            "script",
                        ],
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
                logging_dir=f"{output_path}/{language}/logs_hf",
                job_name="hf_pleias_synth",
                tasks=1,
            )

            hf_executor.run()
