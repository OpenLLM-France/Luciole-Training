from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from slugify import slugify

from datasets import load_dataset_builder

config_names = list(load_dataset_builder("OpenLLM-France/Lucie-Training-Dataset").builder_configs)
print(config_names)

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name", type=str, default="default", help="Subset to load", choices=config_names
    )
    parser.add_argument(
        "--revision", type=str, default="main", help="Revision", choices=["main", "v1.2"]
    )
    args = parser.parse_args()

    name = args.name
    slug_name = slugify(name)
    revision = args.revision
    MAIN_PATH = get_data_path(args.debug, args.local)
    
    pipeline=[ 
        HuggingFaceDatasetReader(
            "OpenLLM-France/Lucie-Training-Dataset",
            {"name": name, "revision": revision, "split": "train"},
            streaming=True
            ),
        JsonlWriter(f"{MAIN_PATH}/lucie_dataset/{slug_name}/output")
    ]

    main_processing_executor = create_pipeline(
        pipeline, slug_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{MAIN_PATH}/lucie_dataset/{slug_name}/logs",
    )

    main_processing_executor.run()