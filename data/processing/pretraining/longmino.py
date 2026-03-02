from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
)

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    LambdaFilter,
)
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def get_source(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
    import re

    for doc in data:
        m = re.search(
            r"hf://datasets/allenai/dolma3_longmino_mix-100B-1125/data/([^/]+)/.*",
            doc.metadata["file_path"],
        )
        doc.metadata["source"] = m.group(1)
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            JsonlReader(
                "hf://datasets/allenai/dolma3_longmino_mix-100B-1125",
                glob_pattern="data/**/*.jsonl.zst",
            ),
            LambdaFilter(
                lambda doc: "lc_synth" in doc.metadata["file_path"]
                or "olmocr_science" in doc.metadata["file_path"]
            ),
            get_source,
            JsonlWriter(
                f"{DATA_PATH}/dolma3_longmino/data",
                output_filename="${source}/${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/dolma3_longmino/logs",
            job_name="longmino",
            tasks=20,
            skip_completed=not args.force,
        )

        main_processing_executor.run()

    else:
        for subset in ["lc_synth-cwe", "lc_synth-rex", "olmocr_science_pdfs"]:
            print(f"\nProcessing {subset}...")

            pipeline = [
                JsonlReader(
                    f"{DATA_PATH}/dolma3_longmino/data",
                    glob_pattern=f"{subset}*/*.jsonl.gz",
                ),
                HuggingFaceDatasetWriter(
                    dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                    + ("-debug" if args.debug else ""),
                    private=True,
                    local_working_dir=f"{DATA_PATH}/dolma3_longmino/{subset}/data_hf",
                    output_filename=f"data/dolma3_longmino/{subset}/en/"
                    + "${rank}.parquet",
                    adapter=partial(
                        _custom_adapter_for_hf,
                        source="dolma3_longmino/" + subset,
                        id_key=None,
                        reset_id=False,
                        language="en",
                        language_key=None,
                        conversation_key=None,
                        remove_keys=["attributes", "no_references"],
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
                logging_dir=f"{DATA_PATH}/dolma3_longmino/{subset}/logs_hf",
                job_name=subset,
                tasks=1,
            )

            hf_executor.run()
