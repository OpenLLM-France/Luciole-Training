from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    print_builder_config,
)

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from slugify import slugify


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
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.name is None:
        print_builder_config("OpenLLM-France/Lucie-Training-Dataset")

    name = args.name
    slug_name = slugify(name)
    revision = args.revision

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
        logging_dir=f"{DATA_PATH}/lucie_dataset/{revision}/{slug_name}/logs",
        job_name=slug_name,
    )

    main_processing_executor.run()
