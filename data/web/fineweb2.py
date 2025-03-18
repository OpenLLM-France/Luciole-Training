import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, MAIN_PATH

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    args = parser.parse_args()
    
    dataset_name="fineweb2"
    language = args.language
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline=[ 
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train", 
            ),
        JsonlWriter(f"{output_path}/data/{language}/train")
    ]

    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        output_path=output_path,
        debug=args.debug,
        local=args.local,
    )

    main_processing_executor.run()