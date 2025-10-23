from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter


class MergeDocument(PipelineStep):
    type = "📑  MERGE"
    name = "📑  Merge Document"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.exclusion_writer = exclusion_writer

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import contextlib
        from datatrove.data import Document

        def breaks_document(chunk, frequency_cutoff, lid_cutoff):
            line = chunk.metadata
            if "None" in [
                line["src_paragraph_id"],
                line["src_sentence_id"],
                line["src_start_id"],
                line["src_end_id"],
                line["tgt_paragraph_id"],
                line["tgt_sentence_id"],
                line["tgt_start_id"],
                line["tgt_end_id"],
            ]:
                return True
            if int(line["duplication_count"]) > frequency_cutoff:
                return True
            if float(line["src_lid_prob"]) < lid_cutoff:
                return True
            if float(line["tgt_lid_prob"]) < lid_cutoff:
                return True
            if len(line["src"].strip()) == 0:
                return True
            if len(line["tgt"].strip()) == 0:
                return True
            return False

        doc = None
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for chunk in data:
                chunk.metadata["src"] = chunk.text
                # Init new doc
                if doc is None:
                    doc = Document(id=chunk.id, text=[], metadata=chunk.metadata)

                # Continue
                if chunk.metadata["src_docid"] == doc.metadata["src_docid"]:
                    if breaks_document(chunk, frequency_cutoff=100, lid_cutoff=0.5):
                        if self.exclusion_writer:
                            writer.write(chunk, rank)
                    else:
                        doc.text.append(
                            {"user": chunk.text, "assistant": chunk.metadata["tgt"]}
                        )

                # break
                else:
                    yield doc
                    doc = None


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "jhu-clsp/paradocs",
            {"name": "en-fr-strict", "split": "train", "trust_remote_code": True},
            streaming=True,
            text_key="src",
        ),
        MergeDocument(
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/paradocs/removed",
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/paradocs/data",
            output_filename="${collection}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/paradocs/logs",
        job_name="paradocs",
        tasks=20,
        partition="prepost",
        time="20:00:00",
        cpu_per_task=4,
    )

    filter_executor.run()
