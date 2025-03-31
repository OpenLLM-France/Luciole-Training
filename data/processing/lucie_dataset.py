from utils import *

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
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision",
        choices=["main", "v1.2"],
    )
    args = parser.parse_args()

    if args.name is None:
        config_names = list(
            load_dataset_builder("OpenLLM-France/Lucie-Training-Dataset").builder_configs
        )
        print(f"Chose a name in: {config_names}")
        raise NotImplementedError

    name = args.name
    slug_name = slugify(name)
    revision = args.revision
    DATA_PATH = get_data_path(args)

    pipeline = [
        HuggingFaceDatasetReader(
            "OpenLLM-France/Lucie-Training-Dataset",
            {"name": name, "revision": revision, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/lucie_dataset/{slug_name}/output"),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/lucie_dataset/{slug_name}/logs",
        job_name=slug_name,
    )

    main_processing_executor.run()
