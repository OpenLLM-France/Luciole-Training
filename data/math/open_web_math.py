import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, MAIN_PATH

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    dataset_name="open_web_math"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline=[ 
        HuggingFaceDatasetReader(
            "open-web-math/open-web-math", 
            {"split": "train"},
            ),
        JsonlWriter(f"{output_path}/output")
    ]

    main_processing_executor = create_pipeline(
        pipeline, 
        dataset_name=dataset_name,
        output_path=output_path,
        debug=args.debug,
        local=args.local,
    )

    main_processing_executor.run()
