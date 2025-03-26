import sys
import os

from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter, FastTextClassifierFilter
from datatrove.data import DocumentsPipeline

# Tried GopherQualityFilter with fineweb-2 config file for french
# It took 7 minutes and 47 secondes for 1k documents...
#     Stats: {total: 538, dropped: 355, dropped_gopher_below_alpha_threshold: 328, dropped_gopher_long_doc: 26, forwarded: 183, doc_len: 17142473 [min=2233, max=592330, 93674.72±96825/doc], dropped_gopher_below_avg_threshold: 1}
        
mapping = {
    'monographies': 'PleIAs/French-PD-Books',
    'press': 'PleIAs/French-PD-Newspapers',
}

def prepare_metadata_header(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    """
        `data` is a generator of Document. You must also return a generator of Document (yield)
        You can optionally use `rank` and `world_size` for sharding
    """
    for doc in data:
        header = "Data extracted from Gallica\n"
        for field, x  in zip(
            ['title', 'author', 'date', 'ocr'], 
            ['Title', 'Author', 'Date', 'OCR quality score']
            ):
            if doc.metadata[field] != 'None':
                header += f"- {x}: {doc.metadata[field]}\n"
        doc.metadata['header'] = header
        yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="monographies",
        choices=list(mapping.keys()),
        help="Dataset to process"
    )
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    hf_name = mapping[args.dataset_name]
    dataset_name = f"gallica_{args.dataset_name}"

    # Collect data and filter OCR by scores
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
        LambdaFilter(
            lambda doc: doc.metadata['author'] not in ['Bourse de Paris']
            ),
        prepare_metadata_header,
        JsonlWriter(f"{output_path}/1_high_ocr_scores")
    ]
    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
    )
    main_processing_executor.run()

    # # FastText tests
    # pipeline=[ 
    #     JsonlReader(f"{output_path}/1_high_ocr_scores"),
    #     FastTextClassifierFilter(
    #         model_url='https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',
    #         keep_labels=('fr', 0.65),
    #         filter_mode="CHUNKS",
    #         save_labels_in_metadata=False,
    #         save_removed_spans=True
    #         ),
    #     JsonlWriter(f"{output_path}/2_fastext_cleaning")
    # ]
    # filtering_executor = create_pipeline(
    #     pipeline, dataset_name,
    #     debug=args.debug,
    #     local=args.local,
    #     logging_dir=f"{output_path}/logs_2",
    # )
    # filtering_executor.run()