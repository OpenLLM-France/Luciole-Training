import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LanguageFilter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name="open_web_math"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline=[ 
        ParquetReader(
            f"hf://datasets/open-web-math/open-web-math/data", 
            ),
        # LanguageFilter(
        #     languages='en',
        #     language_threshold=0.65,
        #     exclusion_writer=JsonlWriter(
        #         f"{output_path}/1_non_english" 
        #         ),
        #     # label_only=True,
        #     ), lot of formulas.. not very effective
        JsonlWriter(f"{output_path}/output")
    ]

    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
    )

    main_processing_executor.run()
