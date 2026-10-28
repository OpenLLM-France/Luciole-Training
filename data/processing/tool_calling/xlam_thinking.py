"""Add reasoning to xlam_oaiformat with a Qwen3 model, keeping only faithful traces.

Step 1 (GPU): for each xlam_oaiformat sample, regenerate the assistant turn with
    thinking enabled. The prompt is the xlam_oaiformat user query, with the
    ``tools`` column passed to vLLM (rendered by the Qwen3 template); the raw
    completion (``<think>...</think><tool_call>...``) is stored in the metadata.
Step 2 (CPU): parse the generated tool calls and keep a sample only if they are
    identical to the ground truth (same calls, order-insensitive). The kept
    assistant turn is the ground-truth tool calls preceded by the generated
    reasoning, converted to the NeMo-RL format and token-counted with the
    hybrid tokenizer like the other tool_calling datasets.
"""

import json
import os
import pathlib
from functools import partial

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter

from utils import (
    create_parser,
    parse_args,
    create_executor,
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
OUT_NAME = "xlam_thinking"


# Commit of OpenLLM-France/tool_data pre-downloaded in the shared HF cache by
# finetune/nemo-rl/download_tool_data.slurm (compute nodes have no internet).
# Needs a commit that has messages_original (5fa6826 does not).
TOOL_DATA_REVISION = "26802821a8652e87f6b8b0bd6f349f102d186410"
DEFAULT_HF_HOME = "/lustre/fsn1/projects/rech/qgz/commun/.cache/huggingface"


def resolve_input_path(revision: str, input_path=None) -> str:
    """Folder holding the xlam_oaiformat parquet files: data/xlam_oaiformat of
    the cached snapshot of OpenLLM-France/tool_data at ``revision``."""
    if input_path:
        return input_path
    hf_home = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    path = (
        f"{hf_home}/hub/datasets--OpenLLM-France--tool_data"
        f"/snapshots/{revision}/data/xlam_oaiformat"
    )
    assert os.path.isdir(path), (
        f"{path} not found: run finetune/nemo-rl/download_tool_data.slurm first "
        "(or pass --input_path / --revision)"
    )
    return path


def _result_text(res):
    text = getattr(res, "text", None)
    if text is None and isinstance(res, dict):
        text = res.get("text")
    return text


def prepare(data, rank: int = 0, world_size: int = 1):
    """Split the xlam_oaiformat messages into the generation context and the
    ground-truth tool calls."""
    for doc in data:
        original = doc.metadata["messages_original"]
        if isinstance(original, str):
            original = json.loads(original)
        tool_calls = original[-1].get("tool_calls")
        if not tool_calls:
            continue
        doc.metadata["ground_truth"] = tool_calls
        tools = doc.metadata["tools"]
        doc.metadata["tool_schemas"] = json.loads(tools) if isinstance(tools, str) else tools
        # Flat messages: [user, assistant] (tools are a separate column). Drop the answer.
        doc.metadata["context"] = [
            {"role": m["role"], "content": m["content"]}
            for m in doc.metadata["messages"][:-1]
        ]
        yield doc


def query_builder(runner, doc, sampling_params: dict):
    return {
        "messages": doc.metadata["context"],
        "tools": doc.metadata["tool_schemas"],
        # "none": tools stay in the prompt, but vLLM returns the raw text
        # instead of requiring --enable-auto-tool-choice + a tool parser.
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling_params,
    }


def postprocess(self, doc, gen_config_meta=None):
    results = doc.metadata.pop("inference_results", None)
    res = results[0] if results else None
    doc.metadata["generation"] = _result_text(res)
    doc.metadata["finish_reason"] = getattr(res, "finish_reason", None)
    if gen_config_meta is not None:
        doc.metadata["generation_config"] = gen_config_meta
    return doc


def _param_types(tools) -> dict:
    """{function name: {parameter name: declared type}} from the tool schemas."""
    if isinstance(tools, str):
        tools = json.loads(tools)
    types = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        properties = (function.get("parameters") or {}).get("properties") or {}
        types[function["name"]] = {
            name: str(prop.get("type", "")) for name, prop in properties.items()
        }
    return types


def _convert_value(value: str, declared_type: str):
    """For the comparison only: XML tool calls carry every argument as text,
    so cast it with the schema type (xlam types look like "str",
    "int, optional", "List[int]"...). Non-string values are returned as is."""
    import ast

    if not isinstance(value, str) or declared_type.lower().startswith(("str", "string")):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        converted = ast.literal_eval(value)  # Python literals: True, None, ['a']...
        json.dumps(converted)  # rejects "...", sets, bytes... (not JSON values)
        return converted
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return value


def _parse_xml_call(block: str):
    """Qwen3.5/3.6 (qwen3_coder) format:
    <function=name>\n<parameter=p>\nvalue\n</parameter>\n</function>
    Values are kept as the raw text Qwen wrote."""
    import re

    match = re.fullmatch(r"<function=([^>\n]+)>(.*?)</function>", block, re.DOTALL)
    if not match:
        return None
    name, body = match.group(1).strip(), match.group(2)
    arguments = {}
    for param, value in re.findall(
        r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.DOTALL
    ):
        arguments[param.strip()] = value
    if re.sub(r"<parameter=.*?</parameter>", "", body, flags=re.DOTALL).strip():
        return None
    return {"name": name, "arguments": arguments}


def parse_generation(text):
    """Return (reasoning, tool_calls) from a raw completion, or None if the
    completion is malformed (unclosed think, invalid tool call...). Tool calls
    can be JSON (Qwen3: {"name": ..., "arguments": ...}) or XML (Qwen3.5/3.6,
    argument values left as text)."""
    import re

    if not text or "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    reasoning = reasoning.replace("<think>", "").strip()
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", answer, re.DOTALL)
    if not blocks:
        return None
    # Nothing but tool calls must follow the reasoning.
    if re.sub(r"<tool_call>.*?</tool_call>", "", answer, flags=re.DOTALL).strip():
        return None
    calls = []
    for block in blocks:
        if block.startswith("<function="):
            call = _parse_xml_call(block)
        else:
            try:
                call = json.loads(block)
            except json.JSONDecodeError:
                return None
        if not isinstance(call, dict) or "name" not in call:
            return None
        calls.append(call)
    return reasoning, calls


def _canonical(call, param_types=None):
    call = call.get("function", call)
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if param_types is not None and isinstance(arguments, dict):
        types = param_types.get(call["name"], {})
        arguments = {k: _convert_value(v, types.get(k, "")) for k, v in arguments.items()}
    return json.dumps(
        {"name": call["name"], "arguments": arguments}, sort_keys=True, ensure_ascii=False
    )


def same_tool_calls(generated, ground_truth, tools=None) -> bool:
    """Order-insensitive equality. Generated argument values are cast with the
    schema types of ``tools`` first (XML tool calls carry them as text)."""
    param_types = _param_types(tools)
    generated = sorted(_canonical(c, param_types) for c in generated)
    return generated == sorted(map(_canonical, ground_truth))


class FilterIdenticalToolCalls(BaseFilter):
    """Keep samples whose generated tool calls equal the ground truth, and
    rebuild the (nested) messages with the generated reasoning; NemoRLFormat
    flattens them afterwards."""

    name = "🎯 Identical tool calls"

    def __init__(self, exclusion_writer=None):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document):
        meta = doc.metadata
        meta.pop("tool_schemas", None)
        if meta.get("finish_reason") != "stop":
            return False, "not_finished"
        parsed = parse_generation(meta.get("generation"))
        if parsed is None:
            return False, "unparsable"
        reasoning, calls = parsed
        if not same_tool_calls(calls, meta["ground_truth"], meta.get("tools")):
            return False, "different_tool_calls"

        # Qwen's reasoning verbatim; its tool calls equal the ground truth
        # (checked above), which carries the properly typed arguments.
        assistant = {
            "role": "assistant",
            "reasoning_content": reasoning,
            "tool_calls": meta["ground_truth"],
        }
        original = meta["messages_original"]
        if isinstance(original, str):
            original = json.loads(original)
        meta["messages"] = original[:-1] + [assistant]
        return True


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--revision", type=str, default=TOOL_DATA_REVISION,
                        help="Commit of OpenLLM-France/tool_data to read xlam_oaiformat from (must be in the HF cache)")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Override: xlam_oaiformat parquet dir (skips the cache lookup)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--model_max_context", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="vLLM --gpu-memory-utilization (default: vLLM's 0.9); "
                             "lower it on unified-memory machines (DGX Spark)")
    args = parse_args(parser)
    DATA_PATH = args.data_path

    input_path = resolve_input_path(args.revision, args.input_path)
    print(f"Reading xlam_oaiformat from {input_path}")

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }
    gen_config_meta = {
        "model": args.model,
        "model_max_context": args.model_max_context,
        "enable_thinking": True,
        "input_dataset": "OpenLLM-France/tool_data",
        "input_revision": args.revision,
        **sampling_params,
    }

    #########
    # Step 1: generate a thinking assistant turn (GPU / vLLM). No server-side
    # tool/reasoning parser: the raw text keeps <think> and <tool_call> blocks,
    # which parse_generation reads in step 2.
    #########
    inference_config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model,
        tp=args.tp,
        temperature=args.temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
        model_kwargs=(
            {"gpu-memory-utilization": args.gpu_memory_utilization}
            if args.gpu_memory_utilization else None
        ),
    )

    generation_pipeline = [
        ParquetReader(input_path, adapter=instruct_adapter),
        prepare,
        InferenceRunner(
            query_builder=partial(query_builder, sampling_params=sampling_params),
            config=inference_config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/{OUT_NAME}/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/generations",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                expand_metadata=True,
            ),
            postprocess_fn=partial(postprocess, gen_config_meta=gen_config_meta),
            skip_bad_requests=True,
        ),
    ]

    generation_executor = create_executor(
        generation_pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs_generation",
        job_name="xlam_thinking_gen",
        tasks=1,
        time="05:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": f"gpu:{args.tp}",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
    )

    #########
    # Step 2: keep only generations whose tool calls match the ground truth (CPU)
    #########
    filter_pipeline = [
        JsonlReader(f"{DATA_PATH}/{OUT_NAME}/generations", adapter=instruct_adapter),
        FilterIdenticalToolCalls(
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/rejected/${{rank}}.jsonl"
            ),
        ),
        NemoRLFormat(),
        check_last_message,
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/{OUT_NAME}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/{OUT_NAME}/hf",
            output_filename=f"data/{OUT_NAME}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
            commit_message=args.commit,
        ),
    ]

    filter_executor = create_executor(
        filter_pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs_filter",
        job_name="xlam_thinking_filter",
        tasks=1,
        time="00:30:00",
        partition="prepost",  # the push to the Hub needs network
        skip_completed=not args.force,
        depends=generation_executor,
    )
    filter_executor.run()
