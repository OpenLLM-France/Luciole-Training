from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from functools import partial
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from typing import Any
import os
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from nemotron_posttraining import convert_messages


def prepare_data(
    data: DocumentsPipeline,
    rank: int = 0,
    world_size: int = 1,
    min_size=200,
    model_name="",
) -> DocumentsPipeline:
    import re
    from datatrove.data import Document

    def extract_thinking(text):
        pattern = r"<think>(.*?)(</think>|$)"
        match = re.search(pattern, text, flags=re.DOTALL)
        extracted = match.group(1).strip() if match else ""
        cleaned = re.sub(pattern, "<think>\n\n</think>", text, flags=re.DOTALL).strip()
        return cleaned, extracted

    def split_min_size(text, min_size=200):
        parts = text.split("\n\n")
        chunks = []
        buffer = ""
        for part in parts:
            if buffer:
                buffer += "\n\n" + part
            else:
                buffer = part
            if len(buffer) >= min_size:
                chunks.append(buffer)
                buffer = ""
        # Add leftover if any
        if buffer:
            chunks.append(buffer)
        return chunks

    for document in data:
        last_turn = document.metadata["messages"][-1]["content"]
        last_turn, thinking = extract_thinking(last_turn)
        chunks = split_min_size(thinking, min_size=min_size)

        document.metadata["num_chunks"] = len(chunks)
        document.metadata["last_turn"] = last_turn
        document.metadata["original_thinking"] = thinking
        document.metadata["model_name"] = model_name

        for i, chunk in enumerate(chunks):
            meta = document.metadata.copy()
            meta["chunk"] = chunk
            meta["chunk_id"] = i

            yield Document(
                id=document.id,
                text=chunk,
                metadata=meta,
            )


def simple_query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
    """
    Simple query builder that extracts text from document for OCR processing.

    Args:
        runner: Inference runner instance
        document: Input document with text content

    Returns:
        Query payload for the inference server
    """
    template = """You are a professional translator. Translate the following text into French. Preserve all mathematical expressions, symbols, and formatting exactly as they appear in the original text. Do not modify any numbers, equations, or formulas. Translate all mathematical commands into first-person plural.

Text:
{text}
"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": template.format(text=document.text)},
                ],
            }
        ],
        "max_tokens": 2048,
    }


class MergeTranslation(PipelineStep):
    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.exclusion_writer = exclusion_writer
        self.buffer = {}

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        from datatrove.data import Document

        for doc in data:
            # Add doc to buffer
            if doc.id not in self.buffer.keys():
                self.buffer[doc.id] = []
            # if doc.metadata["inference_results"][-1].finish_reason == "stop":
            self.buffer[doc.id].append(doc)
            # Check if buffer completed
            if len(self.buffer[doc.id]) == doc.metadata["num_chunks"]:
                completed_list = self.buffer.pop(doc.id)
                completed_list = sorted(
                    completed_list, key=lambda d: d.metadata["chunk_id"]
                )
                merged_document = Document(
                    id=doc.id,
                    text="\n\n".join(
                        [doc.text for doc in completed_list]
                    ),  # sor the list given doc.metadata["chunk_id"] order
                    metadata=completed_list[0].metadata.copy(),
                )
                # Check that all chunks were completed
                if all(
                    d.metadata["inference_results"][0]["finish_reason"] == "stop"
                    for d in completed_list
                ):
                    yield merged_document
                else:
                    if self.exclusion_writer:
                        with self.exclusion_writer as writer:
                            writer.write(merged_document, rank)
        if self.buffer:
            print(f"Buffer not empty. It contains {len(self.buffer)} elements")


def clean_doc(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1, min_size=200
) -> DocumentsPipeline:
    import re

    for doc in data:
        # Remove some fields
        doc.metadata.pop("chunk")
        doc.metadata.pop("chunk_id")
        doc.metadata.pop("inference_results")
        # Postprocess text
        thinking_translated = doc.text
        doc.metadata["thinking_translated"] = thinking_translated
        last_turn = doc.metadata.pop("last_turn")
        pattern = r"<think>(.*?)</think>"

        def repl(_):
            return f"<think>{thinking_translated}</think>"

        last_turn = re.sub(pattern, repl, last_turn, flags=re.DOTALL).strip()
        doc.metadata["messages"][-1]["content"] = last_turn
        yield doc


def sort_chunk_files(files: list[str]) -> list[str]:
    import re

    # pattern: PREFIX_chunk_CHUNK.jsonl.gz
    pattern = re.compile(r"(\d+)_chunk_(\d+)\.jsonl\.gz$")

    def sort_key(filename: str):
        m = pattern.search(filename)
        if not m:
            return (float("inf"), float("inf"))  # put non-matching last
        prefix, chunk = m.groups()
        return (int(prefix), int(chunk))

    return sorted(files, key=sort_key)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--subset",
        type=str,
        default="multilingual_fr",
        help="Subset to process",
        choices=[
            "multilingual_fr",
            "multilingual_es",
            "multilingual_it",
            "multilingual_de",
        ],
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--tp", type=int, default=2)
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "nemotron_posttraining_translation"
    output_path = os.path.join(DATA_PATH, dataset_name, args.subset)

    chat_template = """{%- for message in messages %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
    {%- endfor %}
    {{- '<|im_start|>assistant\n' }}
    {{- '<think>\n\n</think>\n\n' }}"""

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        # temperature=0.6,
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
        chat_template=chat_template,
    )

    def postprocess_fn(self, document):
        document.text = document.metadata["inference_results"][-1].text
        return document

    pipeline = [
        JsonlReader(
            f"/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/nemotron_posttraining/{args.subset}/data",
        ),
        partial(
            prepare_data,
            min_size=1000,
            model_name=args.model_name,
        ),
        InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{output_path}/checkpoints",
            output_writer=JsonlWriter(
                f"{output_path}/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(postprocess_fn),
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    inference_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
        tasks=16,
        time="05:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command="source ~/OpenLLM-BPI-Training/data/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": f"gpu:{args.tp}",
            "nodes": 1,
            "hint": "nomultithread",
        },
    )
    # inference_executor.run()

    # Postprocess
    pipeline = [
        JsonlReader(f"{output_path}/data", order_files=partial(sort_chunk_files)),
        MergeTranslation(
            exclusion_writer=JsonlWriter(
                f"{output_path}/removed_merged",
                output_filename="${rank}.jsonl",
            ),
        ),
        clean_doc,
        partial(
            convert_messages,
            keep_thinking=True,
            language="en"
            if not args.subset.startswith("multilingual")
            else args.subset.split("_")[-1],
        ),
        JsonlWriter(f"{output_path}/data_cleaned", max_file_size=3_221_225_472),
    ]

    final_executor = create_executor(
        pipeline,
        local=args.local,
        debug=False,
        logging_dir=f"{output_path}/logs_cleaned",
        job_name=dataset_name,
        tasks=1,
        time="10:00:00",
        partition="cpu_p1",
        cpus_per_task=2,
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh\nexport HF_HUB_OFFLINE=1",
        # depends=inference_executor,
    )

    final_executor.run()
