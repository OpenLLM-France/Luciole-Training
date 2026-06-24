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
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter

#_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class DPOFilter(BaseFilter):
    name = "✅❌ DPO Preference Filtering"

    def __init__(self, rejected_size, chosen_size, filter_by_length=False, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.filter_by_length = filter_by_length
        self.rejected_size = rejected_size
        self.chosen_size = chosen_size

    def filter(self, doc: Document) -> bool:
        import re

        inference_results_rejected = doc.metadata[f"inference_results_{self.rejected_size}"][0]
        inference_results_chosen = doc.metadata[f"inference_results_{self.chosen_size}"][0]

        if ("text" not in inference_results_chosen) or ("text" not in inference_results_rejected):
            return False, "no_generation"

        doc.metadata["chosen"] = doc.metadata["context"] + [{"role": "assistant", "content": inference_results_chosen["text"]}]
        doc.metadata["rejected"] = doc.metadata["context"] + [{"role": "assistant", "content": inference_results_rejected["text"]}]
        doc.metadata["chosen_size"] = self.chosen_size
        doc.metadata["rejected_size"] = self.rejected_size
        doc.metadata["token_diff"] = inference_results_chosen["usage"]["completion_tokens"] - inference_results_rejected["usage"]["completion_tokens"]
        doc.metadata["token_ratio"] = inference_results_chosen["usage"]["completion_tokens"] / inference_results_rejected["usage"]["completion_tokens"] 
        doc.text = "\n".join([message["content"] for message in doc.metadata["chosen"]])

        if inference_results_rejected["finish_reason"] != "stop" or inference_results_chosen["finish_reason"] != "stop":
            return False, "finish_reason_not_stop"
        
        def normalize(text):
            text = text.lower().strip()
            text = " ".join(text.split())          # normalize whitespace
            text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
            return text

        if normalize(inference_results_chosen["text"]) == normalize(inference_results_rejected["text"]):
            return False, "same_output"

        if "Qwen" in inference_results_chosen["text"]:
            return False, "chosen_contains_qwen_mention"
        
        if re.search(r"[一-鿿]", inference_results_chosen["text"]):
            return False, "chosen_contains_chinese"

        if self.filter_by_length:
            ratio_constraint = doc.metadata["token_ratio"] > 1.3 or doc.metadata["token_ratio"] < (1./1.3)
            diff_constraint = doc.metadata["token_diff"] > 100 or doc.metadata["token_diff"] < -100
            if ratio_constraint and diff_constraint:
                return False, "tokens_ratio_too_large"
        return True


def generation_config(temperature=None, enable_thinking=False, max_tokens=None):
    # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
    # https://huggingface.co/Qwen/Qwen3-32B#best-practices
    if enable_thinking:
        if not temperature:
            temperature = 0.6
        if not max_tokens:
            max_tokens = 32768
        return {
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }    
    else:
        if not temperature:
            temperature = 0.7
        if not max_tokens:
            max_tokens = 2048 
        return {
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": temperature,
            "top_p": 0.80,
            "top_k": 20,
            "min_p": 0.0,
        }


def simple_query_builder(runner, doc, temperature=None, enable_thinking=False, max_tokens=None):
    return {
        "messages": doc.metadata["context"],
        **generation_config(temperature=temperature, enable_thinking=enable_thinking, max_tokens=max_tokens),
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


def postprocess_fn(self, doc, model_size, gen_config):
    doc.metadata[f"inference_results_{model_size}"] = doc.metadata.pop("inference_results")
    doc.metadata[f"generation_config_{model_size}"] = {"model_name_or_path": MODEL_SIZES[model_size], **gen_config}
    return doc


def postprocess_unicode(#TBD
        data, 
        rank: int = 0, 
        world_size: int = 1,
):
    for doc in data:
        for i in range(len(doc.metadata["chosen"])):
            doc.metadata["chosen"][i]["content"] = doc.metadata["chosen"][i]["content"].replace("\u2003", " ")
            doc.metadata["rejected"][i]["content"] = doc.metadata["rejected"][i]["content"].replace("\u2003", " ")
    yield doc

MODEL_SIZES = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1b": "Qwen/Qwen3-1.7B",
    "32b": "Qwen/Qwen3-32B",
}

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input instruct data")
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--rate", type=float, default=0.05, help="Sampling rate")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate (defaults: 32768 thinking, 2048 non-thinking)")
    parser.add_argument("--redo_filtering_only", action="store_true", help="Redo the filtering phase only")
    parser.add_argument("--skip_chosen", action="store_true", help="Skip the chosen generation phase (stage 2), e.g. when the chosen samples have already been generated")
    parser.add_argument("--skip_preproc", action="store_true", help="Skip the prepoc stage if context is already in metadata.")
    parser.add_argument("--chosen_size", type=str, default="32b", choices=MODEL_SIZES.keys(), help="Size of the chosen model")
    parser.add_argument("--rejected_size", type=str, default="1b", choices=MODEL_SIZES.keys(), help="Size of the rejected model")
    args = parse_args(parser)

    config_rejected: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=MODEL_SIZES[args.rejected_size],
        tp=1,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    config_chosen: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=MODEL_SIZES[args.chosen_size],
        tp=2,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    #########
    # Generate rejected samples
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
                f"{args.output_dir}/{args.rejected_size}_sampling/excluded_samples",
                output_filename="${rank}.jsonl",
            ),
        )
    ]
    if not args.skip_preproc:
        pipeline.append(preproc)
    pipeline += [
        InferenceRunner(
            query_builder=partial(simple_query_builder, temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens),
            config=config_rejected,
            records_per_chunk=500,
            checkpoints_local_dir=f"{args.output_dir}/{args.rejected_size}_sampling/checkpoints",
            output_writer=JsonlWriter(
                f"{args.output_dir}/{args.rejected_size}_sampling/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(postprocess_fn, model_size=args.rejected_size, gen_config=generation_config(temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens)),
            skip_bad_requests=True
        ),
    ]

    executor_rejected = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/{args.rejected_size}_sampling/logs",
        job_name=f"{args.rejected_size}_sampling",
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
    # Generate chosen samples
    #########

    executor_chosen = None
    if not args.skip_chosen:
        pipeline = [
            JsonlReader(
                f"{args.output_dir}/{args.rejected_size}_sampling/data",
                adapter=instruct_adapter,
            ),
            InferenceRunner(
                query_builder=partial(simple_query_builder, temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens),
                config=config_chosen,
                records_per_chunk=500,
                checkpoints_local_dir=f"{args.output_dir}/{args.chosen_size}_sampling/checkpoints",
                output_writer=JsonlWriter(
                    f"{args.output_dir}/{args.chosen_size}_sampling/data",
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",
                ),
                postprocess_fn=partial(postprocess_fn, model_size=args.chosen_size, gen_config=generation_config(temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens)),
                skip_bad_requests=True
            ),
        ]

        executor_chosen = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{args.output_dir}/{args.chosen_size}_sampling/logs",
            job_name=f"{args.chosen_size}_sampling",
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
            depends=executor_rejected
        )

    #########
    # Filtering pairs
    #########
    FT176_LANGUAGES = [
        "en",
        "fr",
        "it",
        "de",
        "es",
        "ar",
        "pt",
        "nl",
        "eu",
        "ca",
        "oc",
        "br",
        "co",
        "wa",
    ]
    
    pipeline = [
        JsonlReader(
            f"{args.output_dir}/{args.rejected_size}_sampling/data" if args.skip_chosen else f"{args.output_dir}/{args.chosen_size}_sampling/data",
            adapter=instruct_adapter,
        ),
        DPOFilter(
            rejected_size=args.rejected_size,
            chosen_size=args.chosen_size,
            filter_by_length=False,
            exclusion_writer=JsonlWriter(
                f"{args.output_dir}/filtered_data/excluded_pairs",
                output_filename="${filter_reason}/${rank}.jsonl",
            )
        ),
        LanguageFilter(
            label_only=True,
            keep_top_pairs_threshold=1,
        ),
        LambdaFilter(
            lambda doc: doc.metadata["language"]
            in FT176_LANGUAGES,
            exclusion_writer=JsonlWriter(
                f"{args.output_dir}/filtered_data/excluded_pairs/language_filter",
            ),
        ),
        JsonlWriter(
            f"{args.output_dir}/filtered_data/valid_pairs",
            expand_metadata=True,
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
        qos="qos_cpu-t3",
        skip_completed=not (args.force or args.redo_filtering_only),
        depends=executor_chosen if executor_chosen is not None else executor_rejected
    )
    executor_filtering.run()
