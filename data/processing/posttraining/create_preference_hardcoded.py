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


def generation_config(temperature=0.7):
    return {
        "max_tokens": 2048,
        # turn off reasoning traces for Qwen3
        # chat_template_kwargs": {"enable_thinking": False},
        # Qwen3 recommended non-thinking sampling settings
        # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
        # https://huggingface.co/Qwen/Qwen3-32B#best-practices
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }


def simple_query_builder(runner, doc, temperature=0.7):
    return {
        "messages": doc.metadata["context"],
        **generation_config(temperature=temperature),
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
        doc.metadata["chosen"] = original_messages
        yield doc


def postprocess_fn(self, doc):
    response = doc.metadata["inference_results"].pop()
    doc.metadata["rejected"] = doc.metadata["context"] + [{"role": "assistant", "content": response.text}]
    #doc.metadata[f"generation_config_{model_size}"] = {"model_name_or_path": MODEL_SIZES[model_size], **gen_config}
    return doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input instruct data")
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling (default 0.7 as recommended for Qwen3)")
    parser.add_argument("--rejected_model", type=str, required=True, help="Path to the rejected model")
    parser.add_argument("--rejected_model_name", type=str, required=True, help="name for rejected model")
    args = parse_args(parser)
        
    config_rejected: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.rejected_model,
        tp=1,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )


    #########
    # Generate rejected samples
    #########
    """
    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.glob_pattern,
            adapter=instruct_adapter,
        ),
        SamplerFilter(
            rate=args.rate, seed=42,
            exclusion_writer=JsonlWriter(
                f"{args.output_dir}/{args.rejected_model_name}_sampling/excluded_samples",
                output_filename="${rank}.jsonl",
            ),
        )
    ]"""

    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.glob_pattern,
            adapter=instruct_adapter,
        ),
    ]
    pipeline.append(preproc)
    pipeline += [
        InferenceRunner(
            query_builder=partial(simple_query_builder, temperature=args.temperature),
            config=config_rejected,
            records_per_chunk=500,
            checkpoints_local_dir=f"{args.output_dir}/{args.rejected_model_name}_sampling/checkpoints",
            output_writer=JsonlWriter(
                f"{args.output_dir}/{args.rejected_model_name}_sampling/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                expand_metadata=True,

            ),
            #postprocess_fn=partial(postprocess_fn, model_size=args.rejected_size, gen_config=generation_config(temperature=args.temperature)),
            postprocess_fn=partial(postprocess_fn),
            skip_bad_requests=True
        ),
    ]

    executor_rejected = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/{args.rejected_model_name}_sampling/logs",
        job_name=f"{args.rejected_model_name}_sampling",
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

    executor_rejected.run()

#python create_preference_hardcoded.py --input_path "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/instruct_data/luciole_instruct_sft_mix/hardcoded_en_10_one_to_one.jsonl" --output_dir "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/dpo_data_test/hardcoded_en" --rejected_model "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/sft/luciole_1b/dpo_models/dpo_luciole_latest_mix_v3/huggingface_checkpoints/dpo_luciole_latest_mix_v3-step_2264/" --rejected_model_name "luciole_dpo_1b"
