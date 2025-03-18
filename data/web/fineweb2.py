import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, MAIN_PATH

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

class Rehydrater(PipelineStep):
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        import bisect
        upsampling_weights = {1: 1, 2: 2, 3: 3, 5: 5, 100: 8, 1000: 1}
        # Sorted keys
        limits = sorted(upsampling_weights.keys())

        for doc in data:
            upsampling_weight = upsampling_weights[
                limits[bisect.bisect_right(limits, doc.metadata["minhash_cluster_size"]) - 1]]
            # repeat each document upsampling_weight times
            for _ in range(upsampling_weight):
                yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    args = parser.parse_args()
    
    dataset_name="fineweb2"
    language = args.language

    # Collect data
    output_path = os.path.join(MAIN_PATH, dataset_name)
    pipeline=[ 
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train", 
            ),
        JsonlWriter(f"{output_path}/data/{language}/train")
    ]
    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/train",
    )

    # Rehydrate data
    output_path = os.path.join(MAIN_PATH, dataset_name)
    pipeline=[ 
        JsonlReader(f"{output_path}/data/{language}/train"),
        Rehydrater(),
        JsonlWriter(f"{output_path}/data/{language}/train_upsampled")
    ]
    rehydratation_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/train_upsampled",
        depends=main_processing_executor,
    )
    rehydratation_processing_executor.run()

