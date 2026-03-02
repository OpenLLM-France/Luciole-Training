from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    print_builder_config,
)
import os
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from slugify import slugify
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial
import re
from datatrove.data import DocumentsPipeline

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subset to load",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision",
        choices=["main", "v1.2"],
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default="",
        help="Subset folder in HuggingFace",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.name is None:
        print_builder_config("OpenLLM-France/Lucie-Training-Dataset")

    name = args.name
    slug_name = slugify(name)
    revision = args.revision

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "OpenLLM-France/Lucie-Training-Dataset",
                {"name": name, "revision": revision, "split": "train"},
                streaming=True,
            ),
            JsonlWriter(f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/logs",
            job_name=slug_name,
            cpus_per_task=2,
        )
        main_processing_executor.run()

    else:
        better_slugname = slugify(re.sub(r"(?<!^)(?=[A-Z])", " ", name), separator="_")

        def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
            for doc in data:
                if doc.metadata.get("source") == "CroissantAligned":
                    doc.metadata["language"] = "en-fr"
                else:
                    doc.metadata["language"] = doc.metadata.pop("language")
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/data",
            ),
            fix_data,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/data_hf",
                output_filename=os.path.join(
                    "data",
                    args.hf_subset,
                    better_slugname,
                    "${language}/${rank}.parquet",
                ),
                adapter=partial(
                    _custom_adapter_for_hf,
                    source=better_slugname,
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
            logging_dir=f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/logs_hf",
            job_name=slug_name,
            tasks=1,
            skip_completed=not args.force,
        )

        hf_executor.run()
