import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import create_pipeline, create_parser, MAIN_PATH

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter

class FinewebDocumentCleaning(BaseFilter):

    name = "🧹 FineWeb Document Cleaning"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        log_repeated_lines: bool = False,
    ):
        super().__init__(exclusion_writer)
        self.log_repeated_lines = log_repeated_lines

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lines = doc.text.strip().splitlines()
        if self.log_repeated_lines:
            doc.metadata['repeated_line'] = False
        # Iterating on lines
        unique_lines = []
        for line in lines:
            if "|" in line: # | is a separator in markdown tables
                unique_lines.append(line)
            elif not unique_lines or line != unique_lines[-1]:
                unique_lines.append(line)
            else:
                self.stat_update("repeated_line")
                if self.log_repeated_lines:
                    doc.metadata['repeated_line'] = True
        doc.text = '\n'.join(unique_lines)
        # Check if the document is empty after cleaning
        if not doc.text.strip():
            return False
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
    
    dataset_name="fineweb2"
    language = args.language

    # Collect data
    output_path = os.path.join(MAIN_PATH, dataset_name)
    pipeline=[ 
        ParquetReader(
            f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train", 
            ),
        FinewebDocumentCleaning(),
        Rehydrater(),
        JsonlWriter(
            f"{output_path}/data/{language}/clusters",
            output_filename="${cluster_size_group}/${rank}.jsonl.gz", 
        )
    ]
    main_processing_executor = create_pipeline(
        pipeline, dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs/{language}/train",
    )
    main_processing_executor.run()

