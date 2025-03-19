import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, MAIN_PATH

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter

# Tried GopherQualityFilter with fineweb-2 config file for french
# It took 7 minutes and 47 secondes for 1k documents...
#     Stats: {total: 538, dropped: 355, dropped_gopher_below_alpha_threshold: 328, dropped_gopher_long_doc: 26, forwarded: 183, doc_len: 17142473 [min=2233, max=592330, 93674.72±96825/doc], dropped_gopher_below_avg_threshold: 1}
        
mapping = {
    'monographies': 'PleIAs/French-PD-Books',
    'press': 'PleIAs/French-PD-Newspapers',
}

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="monographies",
        choices=list(mapping.keys()),
        help="Dataset to process"
    )
    args = parser.parse_args()
    
    hf_name = mapping[args.dataset_name]
    dataset_name = f"gallica_{args.dataset_name}"

    # Collect data
    output_path = os.path.join(MAIN_PATH, dataset_name)
    pipeline=[ 
        ParquetReader(
            f"hf://datasets/{hf_name}", 
            glob_pattern = "*.parquet",
            text_key="complete_text"
            ),
        LambdaFilter(
            lambda doc: int(doc.metadata['ocr']) >= 90,
            exclusion_writer=JsonlWriter(
                f"{output_path}/1_low_ocr_scores" 
                )
            ),
        JsonlWriter(f"{output_path}/1_high_ocr_scores")
    ]
    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
    )
    main_processing_executor.run()

