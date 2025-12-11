from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter


def merge_document(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from datatrove.data import Document

    doc = None
    for chunk in data:
        if doc is None:
            doc = Document(
                id=chunk.metadata["src_docid"],
                text=[chunk.text],
                metadata={
                    "tgt": [chunk.metadata["tgt"]],
                    "collection": chunk.metadata["collection"],
                },
            )
        elif doc.id == chunk.metadata["src_docid"]:
            # Add chunk to current document
            doc.text.append(chunk.text)
            doc.metadata["tgt"].append(chunk.metadata["tgt"])
        else:
            yield doc
            doc = Document(
                id=chunk.metadata["src_docid"],
                text=[chunk.text],
                metadata={
                    "tgt": [chunk.metadata["tgt"]],
                    "collection": chunk.metadata["collection"],
                },
            )
    yield doc


class ParadocsProcessing(PipelineStep):
    def __init__(
        self,
        src_language,
        tgt_language,
        revert_prob=0.5,
        geometric_param=1.0 / 3,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.revert_prob = revert_prob
        self.geometric_param = geometric_param
        self.exclusion_writer = exclusion_writer

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import random

        def chunk_parallel_lists(src, tgt):
            import numpy as np

            assert len(src) == len(tgt), "src and tgt must have the same length"

            src_chunks, tgt_chunks = [], []
            i = 0
            n = len(src)

            while i < n:
                # choose random chunk length between min_size and max_size
                chunk_size = np.random.geometric(self.geometric_param, size=None)
                src_chunks.append(src[i : i + chunk_size])
                tgt_chunks.append(tgt[i : i + chunk_size])
                i += chunk_size

            src_chunks = ["\n".join(s) for s in src_chunks]
            tgt_chunks = ["\n".join(s) for s in tgt_chunks]
            return src_chunks, tgt_chunks

        for doc in data:
            do_revert = random.random() < self.revert_prob
            if not do_revert:
                src = doc.text
                tgt = doc.metadata.pop("tgt")
                doc.metadata["src_language"] = self.src_language
                doc.metadata["tgt_language"] = self.tgt_language
            else:
                src = doc.metadata.pop("tgt")
                tgt = doc.text
                doc.metadata["src_language"] = self.tgt_language
                doc.metadata["tgt_language"] = self.src_language
            doc.text = " "  # Clear text

            # Merge chunk
            src, tgt = chunk_parallel_lists(src, tgt)

            # Conversation template
            doc.metadata["conversation"] = []
            # num_of_chunk_in_current_turn = np.random.poisson(3)
            for user_turn, assistant_turn in zip(src, tgt):
                doc.metadata["conversation"].extend(
                    [
                        {"role": "user", "content": user_turn},
                        {"role": "assistant", "content": assistant_turn},
                    ]
                )
            yield doc


def apply_template(
    data: DocumentsPipeline,
    rank: int = 0,
    world_size: int = 1,
) -> DocumentsPipeline:
    for doc in data:
        conversation = doc.metadata["conversation"]
        doc.text = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "user":
                doc.text += "- " + doc.metadata["src_language"] + ": " + content + "\n"
            elif role == "assistant":
                doc.text += (
                    "- " + doc.metadata["tgt_language"] + ": " + content + "\n\n"
                )
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--languages", default="en-fr", type=str)
    parser.add_argument("--revert", default=0.5, type=float)
    parser.add_argument("--geometric_param", default=1.0 / 3, type=float)
    args = parse_args(parser)
    src_language, tgt_language = args.languages.split("-")
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "jhu-clsp/paradocs",
            {
                "name": f"{args.languages}-strict",
                "split": "train",
                "trust_remote_code": True,
            },
            streaming=True,
            text_key="src",
        ),
        merge_document,
        ParadocsProcessing(
            src_language=src_language,
            tgt_language=tgt_language,
            revert_prob=args.revert,
        ),
        apply_template,
        JsonlWriter(
            f"{DATA_PATH}/paradocs_geom{args.geometric_param:.2f}/{args.languages}_revert{args.revert}/data",
            output_filename="${collection}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/paradocs_geom{args.geometric_param:.2f}/{args.languages}_revert{args.revert}/logs",
        job_name="paradocs",
        tasks=20,
        partition="prepost",
        time="20:00:00",
        cpu_per_task=4,
    )

    filter_executor.run()

# python paradocs.py --languages en-fr --revert 0.5
# python paradocs.py --languages en-es --revert 0.5
# python paradocs.py --languages en-nl --revert 0.5
# python paradocs.py --languages en-de --revert 0.5
# python paradocs.py --languages en-pt --revert 0.5
# python paradocs.py --languages en-it --revert 0.5
