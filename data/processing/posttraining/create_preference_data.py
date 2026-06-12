# Script that input instruct data and output preference data for posttraining

import os
import pathlib
from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from functools import partial
from utils import instruct_adapter
from datatrove.pipeline.filters import SamplerFilter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

#_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class DPOFilter(BaseFilter):
    name = "✅❌ DPO Preference Filtering"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool:
        import re

        inference_results_1b = doc.metadata["inference_results_1b"][0]
        inference_results_32b = doc.metadata["inference_results_32b"][0]

        # Format
        doc.metadata["chosen"] = doc.metadata["context"] + [{"role": "assistant", "content": inference_results_32b["text"]}]
        doc.metadata["rejected"] = doc.metadata["context"] + [{"role": "assistant", "content": inference_results_1b["text"]}]

        # Filter
        if inference_results_1b["finish_reason"] != "stop" or inference_results_32b["finish_reason"] != "stop":
            return False, "finish_reason_not_stop"

        if "Qwen" in inference_results_32b["text"]:
            return False, "chosen_contains_qwen_mention"
        
        if re.search(r"[一-鿿]", inference_results_32b["text"]):
            return False, "chosen_contains_chinese"

        doc.metadata["token_diff"] = inference_results_32b["usage"]["completion_tokens"] - inference_results_1b["usage"]["completion_tokens"]
        doc.metadata["token_ratio"] = inference_results_32b["usage"]["completion_tokens"] / inference_results_1b["usage"]["completion_tokens"] 
        if doc.metadata["token_ratio"] > 1.3 or doc.metadata["token_ratio"] < (1./1.3) :
            return False, "tokens_ratio_too_large"
        return True


def simple_query_builder(runner, doc, temperature=0.7):
    return {
        "messages": doc.metadata["context"],
        "max_tokens": 2048,
        # turn off reasoning traces for Qwen3
        "chat_template_kwargs": {"enable_thinking": False},
        # Qwen3 recommended non-thinking sampling settings 
        # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
        # https://huggingface.co/Qwen/Qwen3-32B#best-practices
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }


def preproc(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        assert "messages" in doc.metadata, "Each document must have 'messages' in metadata"
        original_messages = doc.metadata.pop("messages")
        idx = max(i for i, m in enumerate(original_messages) if m["role"] == "assistant")  # Index of last assistant message
        doc.metadata["context"] = original_messages[:idx] # Keep all messages before the last assistant message as context
        doc.metadata["original_response"] = original_messages[idx:] # Keep the last assistant message and all messages after as response
        yield doc


def postprocess_fn(self, doc, model_size):
    doc.metadata[f"inference_results_{model_size}"] = doc.metadata.pop("inference_results")
    return doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input instruct data")
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--rate", type=float, default=0.05, help="Sampling rate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default 0.7 as recommended for Qwen3)")
    args = parse_args(parser)

    config_1b: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path="Qwen/Qwen3-1.7B",
        tp=1,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    config_32b: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path="Qwen/Qwen3-32B",
        tp=2,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    #########
    # Generate 1B rejected samples
    #########

    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.glob_pattern,
            adapter=instruct_adapter,
        ),
        SamplerFilter(
            rate=args.rate, seed=42,
            exclusion_writer=JsonlWriter(
                f"{args.output_dir}/1b_sampling/excluded_samples",
                output_filename="${rank}.jsonl",
            ),
        ),
        preproc,
        InferenceRunner(
            query_builder=partial(simple_query_builder, temperature=args.temperature),
            config=config_1b,
            records_per_chunk=500,
            checkpoints_local_dir=f"{args.output_dir}/1b_sampling/checkpoints",
            output_writer=JsonlWriter(
                f"{args.output_dir}/1b_sampling/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(postprocess_fn, model_size="1b"),
        ),
    ]

    executor_1b = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/1b_sampling/logs",
        job_name="1b_sampling",
        tasks=1,
        time="10:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        #env_command="source ~/OpenLLM-BPI-Training/data/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": "gpu:1",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
    )

    #########
    # Generate 32B accepted samples
    #########

    pipeline = [
        JsonlReader(
            f"{args.output_dir}/1b_sampling/data",
            adapter=instruct_adapter,
        ),
        InferenceRunner(
            query_builder=simple_query_builder,
            config=config_32b,
            records_per_chunk=500,
            checkpoints_local_dir=f"{args.output_dir}/32b_sampling/checkpoints",
            output_writer=JsonlWriter(
                f"{args.output_dir}/32b_sampling/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(postprocess_fn, model_size="32b"),
        ),
    ]

    executor_32b = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/32b_sampling/logs",
        job_name="32b_sampling",
        tasks=1,
        time="10:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        #env_command="source ~/OpenLLM-BPI-Training/data/set_env_inference.sh",
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": "gpu:2",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
        depends=executor_1b
    )

    #########
    # Filtering pairs
    #########

    pipeline = [
        JsonlReader(
            f"{args.output_dir}/32b_sampling/data",
            adapter=instruct_adapter,
        ),
        DPOFilter(
            exclusion_writer=JsonlWriter(
                f"{args.output_dir}/filtered_data/excluded_pairs",
                output_filename="${rank}.jsonl",
            )
        ),
        JsonlWriter(
            f"{args.output_dir}/filtered_data/valid_pairs",
        ),
    ]

    executor_filtering = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/filtered_data/logs",
        job_name="dpo_filter",
        tasks=1,
        time="02:00:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
        depends=executor_32b
    )
    executor_filtering.run()
