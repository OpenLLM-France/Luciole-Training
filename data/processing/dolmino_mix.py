from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from slugify import slugify
from datasets import load_dataset_builder

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
        config_names = list(
            load_dataset_builder("allenai/dolmino-mix-1124").builder_configs
        )
        print(f"Chose a name in: {config_names}")
        raise NotImplementedError

    name = args.name
    slug_name = slugify(name)

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/dolmino-mix-1124",
            {"name": {name}, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/dolmino_mix/{slug_name}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/dolmino_mix/{slug_name}/logs",
        job_name=slug_name,
    )

    main_processing_executor.run()
