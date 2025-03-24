import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters import RegexFilter

class FinewebDocumentCleaning(BaseFilter):

    name = "🧹 FineWeb Document Cleaning"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        ## Line removal
        lines = doc.text.strip().splitlines()
        kept_lines = []
        for i, line in enumerate(lines, 0):
            if len(lines) > 1:
                if i == 0 and (line in lines[1]):
                    self.stat_update("first-line-included-in-second")
                    continue
                if i == 1 and (line in lines[0]):
                    self.stat_update("second-line-included-in-first")
                    continue
            if kept_lines and (line != '') and ('|' not in line) and (line == kept_lines[-1]):
                self.stat_update("consecutive-repeated-lines")
                continue
            kept_lines.append(line)
            self.stat_update("line-kept")
        doc.text = '\n'.join(kept_lines)
        return True

class Rehydrater(PipelineStep):
    @staticmethod
    def get_cluster_size_group(cluster_size):
        if cluster_size in [1, 2, 3, 4]:
            return f'cluster_size:{cluster_size}'
        elif cluster_size < 100:
            return 'cluster_size:5-100'
        elif cluster_size < 1000:
            return 'cluster_size:100-1000'
        else:
            return 'cluster_size:1000+'
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            cluster_size_group = self.get_cluster_size_group(doc.metadata["minhash_cluster_size"])
            doc.metadata["cluster_size_group"] = cluster_size_group
            yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name="fineweb2"
    language = args.language
    output_path = os.path.join(MAIN_PATH, dataset_name)

    ## Collect data
    pipeline=[ 
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train", 
            ),
        FinewebDocumentCleaning(),
        JsonlWriter(
            f"{output_path}/data/{language}/train",        
            )
    ]
    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/train",
    )
    main_processing_executor.run()

    ## Split by clusters
    pipeline=[ 
        JsonlReader(
            f"{output_path}/data/{language}/train"
        ),
        Rehydrater(),
        JsonlWriter(
            f"{output_path}/data/{language}/clusters",
            output_filename="${cluster_size_group}/${rank}.jsonl.gz", 
        )
    ]
    split_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/clusters",
        depends_on=main_processing_executor,
    )
    split_executor.run()

    ## Extract potential copyrights - there are not remove from the data!
    pipeline=[ 
        JsonlReader(
            f"{output_path}/data/{language}/train"
        ),
        RegexFilter(
            regex_exp=r"(Copyright|copyright|©|All\s+rights\s+reserved)",
            exclusion_writer=JsonlWriter(
                f"{output_path}/data/{language}/potential_copyrights" 
                )
        )
    ]
    copyright_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/clusters",
        depends_on=main_processing_executor,
    )
    copyright_executor.run()