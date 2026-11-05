"""Regenerate the xlam_oaiformat answers with a Qwen model (thinking or not), keeping only faithful ones.

Step 1 (GPU): for each xlam_oaiformat sample, regenerate the assistant turn (with
    ``--enable_thinking``: reasoning + calls). The prompt is the xlam_oaiformat user query, with the
    ``tools`` column passed to vLLM (rendered by the Qwen3 template); the raw
    completion (``<think>...</think><tool_call>...``) is stored in the metadata.
Step 2 (CPU): parse the generated tool calls and keep a sample only if they are
    identical to the ground truth (same calls, order-insensitive; text before
    the calls is allowed, text after them is not). The kept assistant turn is
    the ground-truth tool calls preceded by the generated reasoning and text,
    converted to the NeMo-RL format and token-counted with the hybrid
    tokenizer like the other tool_calling datasets.

Perturbation (``--remove_tool_ratio``, default 10% of the prompts): for that ratio of the
prompts, one tool the ground truth calls is removed from the tools, and its
calls from the ground truth (see remove_tool). If no call is left, the kept
answer is the generated text (any tool call is rejected).
``--only_removed_tool`` generates only those perturbed prompts.

Thinking is off unless ``--enable_thinking`` (same flag and sampling settings
as xlam_dpo, see generation_config); non-thinking answers are kept without
reasoning, in ``xlam_non_thinking[_removed_tool]`` instead of
``xlam_thinking[_removed_tool]``.
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
OUT_NAME = "xlam_thinking"  # set from the args in __main__


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


def generation_config(temperature=None, enable_thinking=False, max_tokens=None):
    # Same as posttraining/create_preference_data.py, except max_tokens with
    # thinking: 4096 instead of 32768 (xlam reasoning is short, ~1k characters).
    # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
    # https://huggingface.co/Qwen/Qwen3-32B#best-practices
    if enable_thinking:
        if not temperature:
            temperature = 0.6
        if not max_tokens:
            max_tokens = 4096
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


def xlam_sft_hub_adapter(self, document):
    """hub_adapter plus the generation settings, like xlam_dpo keeps them:
    ``generation_config`` (JSON string: model, sampling, input revision,
    remove_tool_ratio) and ``removed_tool`` ("" when no tool was removed)."""
    row = hub_adapter(self, document)
    config = document.metadata.get("generation_config") or {}
    row["generation_config"] = (
        config if isinstance(config, str) else json.dumps(config, ensure_ascii=False)
    )
    row["removed_tool"] = str(document.metadata.get("removed_tool") or "")
    return row


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


def remove_tool(data, rank: int = 0, world_size: int = 1, ratio: float = 0.1, seed: int = 0):
    """Perturbation: for a ``ratio`` of the prompts, remove one tool that the
    ground truth calls (picked at random) from the tool definitions, and drop
    its calls from the ground truth. When no call is left, the expected answer
    is a text (ground_truth == []). ``metadata["removed_tool"]`` is the removed
    tool name (None if unperturbed).

    The draw is seeded per prompt with ``seed`` and the prompt id, not with one
    global RNG: a global RNG would depend on the reading order, which changes
    with the number of tasks and when datatrove resumes from checkpoints.
    ``seed`` picks which prompts / tools: use a different one per dataset so
    their perturbed prompts differ. ``seed=0`` is the draw of the first runs
    (xlam_thinking with Qwen3.5-35B-A3B, xlam_irrelevance, the xlam_dpo data).
    """
    import random

    for doc in data:
        meta = doc.metadata
        meta["removed_tool"] = None
        rng = random.Random(f"remove_tool-{doc.id}" if seed == 0 else f"remove_tool-{seed}-{doc.id}")
        if rng.random() < ratio:
            defined = {t.get("function", t)["name"] for t in meta["tool_schemas"]}
            used = sorted(
                {c.get("function", c)["name"] for c in meta["ground_truth"]} & defined
            )
            if used:
                name = rng.choice(used)
                meta["tool_schemas"] = [
                    t for t in meta["tool_schemas"] if t.get("function", t)["name"] != name
                ]
                meta["tools"] = json.dumps(meta["tool_schemas"], ensure_ascii=False)
                meta["ground_truth"] = [
                    c for c in meta["ground_truth"] if c.get("function", c)["name"] != name
                ]
                meta["removed_tool"] = name
        yield doc


def only_removed_tool(data, rank: int = 0, world_size: int = 1):
    """Keep only the prompts perturbed by remove_tool (--only_removed_tool)."""
    for doc in data:
        if doc.metadata.get("removed_tool"):
            yield doc


def query_builder(runner, doc, gen_config: dict):
    payload = {
        "messages": doc.metadata["context"],
        **gen_config,
    }
    if doc.metadata["tool_schemas"]:  # no tools left after remove_tool: plain chat request
        payload["tools"] = doc.metadata["tool_schemas"]
        # "none": tools stay in the prompt, but vLLM returns the raw text
        # instead of requiring --enable-auto-tool-choice + a tool parser.
        payload["tool_choice"] = "none"
    return payload


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
    """Return (reasoning, content, tool_calls) from a raw completion, or None
    if the completion is malformed (unclosed think, invalid tool call, text
    after the calls...). ``content`` is the text written before the calls
    (None if there is none): it is allowed, only the calls are checked. Tool
    calls can be JSON (Qwen3: {"name": ..., "arguments": ...}) or XML
    (Qwen3.5/3.6, argument values left as text)."""
    import re

    if not text or "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    reasoning = reasoning.replace("<think>", "").strip()
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", answer, re.DOTALL)
    if not blocks:
        return None
    # Text may come before the calls (the content), but nothing else than
    # tool calls after the first one.
    content, calls_part = answer.split("<tool_call>", 1)
    if re.sub(r"<tool_call>.*?</tool_call>", "", "<tool_call>" + calls_part, flags=re.DOTALL).strip():
        return None
    if "<think>" in content or "</think>" in content:  # a second reasoning block
        return None
    content = content.strip() or None
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
    return reasoning, content, calls


def parse_text_answer(text):
    """Return (reasoning, content) of a completion that answers with text only
    (no tool call), or None (unclosed think, empty answer, tool call markup)."""
    if not text or "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    answer = answer.strip()
    if not answer or "<tool_call>" in answer or "<function=" in answer:
        return None
    return reasoning.replace("<think>", "").strip(), answer


def is_text_after_calls(text) -> bool:
    """True if the answer has complete tool calls followed by text: the
    content must come before the calls, and text written after them typically
    claims a result the tools have not returned yet."""
    if not text or "</think>" not in text:
        return False
    answer = text.split("</think>", 1)[1]
    last = answer.rfind("</tool_call>")
    return "<tool_call>" in answer and last != -1 and bool(answer[last + len("</tool_call>"):].strip())


def _normalize_numbers(value):
    """Integral floats as ints (2000.0 -> 2000), so that numbers compare by
    value: the JSON text of 2000 and 2000.0 differs."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_numbers(v) for k, v in value.items()}
    return value


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
    arguments = _normalize_numbers(arguments)
    return json.dumps(
        {"name": call["name"], "arguments": arguments}, sort_keys=True, ensure_ascii=False
    )


def same_tool_calls(generated, ground_truth, tools=None) -> bool:
    """Order-insensitive equality. Generated argument values are cast with the
    schema types of ``tools`` first (XML tool calls carry them as text), and
    numbers compare by value (2000 == 2000.0)."""
    param_types = _param_types(tools)
    generated = sorted(_canonical(c, param_types) for c in generated)
    return generated == sorted(map(_canonical, ground_truth))


class FilterIdenticalToolCalls(BaseFilter):
    """Keep samples whose generated tool calls equal the ground truth, and
    rebuild the (nested) messages with the generated reasoning; NemoRLFormat
    flattens them afterwards."""

    name = "🎯 Identical tool calls"

    def __init__(self, exclusion_writer=None, enable_thinking: bool = True):
        super().__init__(exclusion_writer)
        self.enable_thinking = enable_thinking

    def filter(self, doc: Document):
        meta = doc.metadata
        meta.pop("tool_schemas", None)
        if meta.get("finish_reason") != "stop":
            return False, "not_finished"
        generation = meta.get("generation")
        if not self.enable_thinking and generation is not None:
            # The template puts the empty think block in the prompt, so the
            # completion is just the answer: close an (empty) reasoning for
            # the parsers. A <think> written anyway is rejected below.
            if "<think>" in generation or "</think>" in generation:
                return False, "unparsable"
            generation = "</think>" + generation
        parsed = parse_generation(generation)
        if not meta["ground_truth"]:
            # Every called tool was removed (see remove_tool): a text answer
            # is expected, any tool call is wrong.
            if parsed is not None:
                return False, "irrelevance_error"
            text_answer = parse_text_answer(generation)
            if text_answer is None:
                return False, "unparsable"
            reasoning, content = text_answer
            assistant = {"role": "assistant", "content": content}
        else:
            if parsed is None:
                if is_text_after_calls(generation):
                    return False, "text_after_call"
                if parse_text_answer(generation) is not None:
                    return False, "relevance_error"
                return False, "unparsable"
            reasoning, content, calls = parsed
            if not same_tool_calls(calls, meta["ground_truth"], meta.get("tools")):
                return False, "different_tool_calls"

            # Qwen's reasoning and text before the calls verbatim; its tool
            # calls equal the ground truth (checked above), which carries the
            # properly typed arguments (minus the removed tool's calls).
            assistant = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": meta["ground_truth"],
            }
        if self.enable_thinking:
            assistant["reasoning_content"] = reasoning
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
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling (defaults: 0.6 thinking, 0.7 non-thinking)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate (defaults: 4096 thinking, 2048 non-thinking)")
    parser.add_argument("--model_max_context", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="vLLM --gpu-memory-utilization (default: vLLM's 0.9); "
                             "lower it on unified-memory machines (DGX Spark)")
    parser.add_argument("--remove_tool_ratio", type=float, default=0.1,
                        help="Ratio of prompts where one called tool is removed from the tools "
                             "and its calls from the ground truth (text answer if none is left)")
    # Seed of the remove_tool draw (which prompts lose which tool), salted with
    # the prompt id. Default 0 here and 1 in xlam_dpo, so the SFT and DPO data
    # perturb different prompts. 0 reproduces data/sft/xlam_thinking (tool_data
    # b9be666e); use another seed for the --only_removed_tool data (e.g.
    # xlam_irrelevance: --seed 2) so its prompts are not those of xlam_thinking.
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the remove_tool draw (salted with the prompt id); use a "
                             "different seed per dataset (xlam_dpo default: 1)")
    parser.add_argument("--only_removed_tool", action="store_true",
                        help="Only generate the prompts perturbed by --remove_tool_ratio")
    args = parse_args(parser)
    DATA_PATH = args.data_path
    enable_thinking = args.enable_thinking
    assert not args.only_removed_tool or args.remove_tool_ratio > 0, \
        "--only_removed_tool needs --remove_tool_ratio > 0"
    OUT_NAME = "xlam_thinking" if enable_thinking else "xlam_non_thinking"
    if args.only_removed_tool:
        OUT_NAME += "_removed_tool"

    input_path = resolve_input_path(args.revision, args.input_path)
    print(f"Reading xlam_oaiformat from {input_path}")

    gen_config = generation_config(
        temperature=args.temperature, enable_thinking=enable_thinking, max_tokens=args.max_tokens
    )
    gen_config_meta = {
        "model_name_or_path": args.model,
        **gen_config,
        "model_max_context": args.model_max_context,
        "remove_tool_ratio": args.remove_tool_ratio,
        "remove_tool_seed": args.seed,
        "only_removed_tool": args.only_removed_tool,
        "input_dataset": "OpenLLM-France/tool_data",
        "input_revision": args.revision,
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
        temperature=gen_config["temperature"],
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
        partial(remove_tool, ratio=args.remove_tool_ratio, seed=args.seed),
        *([only_removed_tool] if args.only_removed_tool else []),
        InferenceRunner(
            query_builder=partial(query_builder, gen_config=gen_config),
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
        job_name=f"{OUT_NAME}_gen",
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
            enable_thinking=enable_thinking,
        ),
        NemoRLFormat(),
        check_last_message,
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": enable_thinking, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/{OUT_NAME}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/{OUT_NAME}/hf",
            output_filename=f"data/sft/{OUT_NAME}/${{rank}}.parquet",
            adapter=xlam_sft_hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            commit_message=args.commit,
        ),
    ]

    filter_executor = create_executor(
        filter_pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs_filter",
        job_name=f"{OUT_NAME}_filter",
        tasks=1,
        time="00:30:00",
        partition="prepost",  # the push to the Hub needs network
        skip_completed=not args.force,
        depends=generation_executor,
    )
    filter_executor.run()
