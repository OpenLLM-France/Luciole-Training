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
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.name is None:
        print_builder_config("allenai/olmo-mix-1124")

    name = args.name
    slug_name = slugify(name)

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/olmo-mix-1124",
            {"name": {name}, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/olmo_mix/{slug_name}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/olmo_mix/{slug_name}/logs",
        job_name=slug_name,
    )

    main_processing_executor.run()
