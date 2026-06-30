import pathlib
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from functools import partial
from transformers import AutoTokenizer
from utils import (
    FilterChinese,
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
    nemo_rl_format_messages,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


def create_rejected(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import copy
    import json
    import random

    def _tool_name(tool_call):
        src = tool_call.get("function", tool_call)
        return src["name"]

    def _tool_args(tool_call):
        src = tool_call.get("function", tool_call)
        return src.get("arguments", {}) or {}

    def _required_map(tools):
        """Map each tool name to the set of its required parameters."""
        required = {}
        for tool in tools:
            fn = tool.get("function", tool)
            params = fn.get("parameters", {}) or {}
            required[fn["name"]] = set(params.get("required", []) or [])
        return required

    # --- Independent corruption functions ----------------------------------
    # Each takes the chosen assistant message and the parsed tool schemas and
    # returns ``(success, rejected)``: ``success`` says whether the corruption
    # applies to this example (e.g. there is more than one call to remove), and
    # ``rejected`` is the corrupted assistant message (or None when it is
    # produced by a later step, as for corrupt_argument).

    def remove_tool_call(chosen_answer, required_map):
        # if tool_calls > 1, remove a random subset of the tools (keep at
        # least 1, remove at least 1).
        tool_calls = chosen_answer.get("tool_calls") or []
        if len(tool_calls) < 2:
            return False, None
        n_keep = random.randint(1, len(tool_calls) - 1)
        kept = sorted(random.sample(range(len(tool_calls)), n_keep))
        rejected = copy.deepcopy(chosen_answer)
        rejected["tool_calls"] = [tool_calls[i] for i in kept]
        return True, rejected

    def remove_required_argument(chosen_answer, required_map):
        # Remove a random number (1 to all) of the required arguments that are
        # present across the tool calls.
        tool_calls = chosen_answer.get("tool_calls") or []
        candidates = []
        for call_idx, tool_call in enumerate(tool_calls):
            required = required_map.get(_tool_name(tool_call), set())
            for arg in _tool_args(tool_call):
                if arg in required:
                    candidates.append((call_idx, arg))
        if not candidates:
            return False, None
        n_remove = random.randint(1, len(candidates))
        to_remove = random.sample(candidates, n_remove)
        rejected = copy.deepcopy(chosen_answer)
        for call_idx, arg in to_remove:
            _tool_args(rejected["tool_calls"][call_idx]).pop(arg, None)
        return True, rejected

    def corrupt_argument(chosen_answer, required_map):
        # Deferred to a downstream LLM step: only decide here whether the
        # corruption applies (there is at least one argument to corrupt) and
        # leave the rejected answer to be filled in later.
        tool_calls = chosen_answer.get("tool_calls") or []
        if not any(_tool_args(tool_call) for tool_call in tool_calls):
            return False, None
        return True, None

    def corrupt_json(chosen_answer, required_map):
        # Render the tool calls and break the JSON of one of them so the call
        # is no longer parseable. The corrupted render is stored as plain
        # content (no structured tool_calls) so it survives the chat template.
        tool_calls = chosen_answer.get("tool_calls") or []
        if not tool_calls:
            return False, None

        def break_json(text):
            options = []
            if "}" in text:
                idx = text.rfind("}")
                options.append(text[:idx] + text[idx + 1:])  # drop a closing brace
            if '"' in text:
                options.append(text.replace('"', "", 1))  # drop a quote
            if ":" in text:
                options.append(text.replace(":", "", 1))  # drop a colon
            options.append(text + ",")  # trailing junk
            broken = random.choice(options)
            try:
                json.loads(broken)
            except Exception:
                return broken
            # Guarantee invalidity if the chosen mutation happened to parse.
            return text.rstrip("}") or (text + "{")

        target = random.randrange(len(tool_calls))
        blocks = []
        for i, tool_call in enumerate(tool_calls):
            obj = json.dumps(
                {"name": _tool_name(tool_call), "arguments": _tool_args(tool_call)}
            )
            if i == target:
                obj = break_json(obj)
            blocks.append(f"<tool_call>\n{obj}\n</tool_call>")
        return True, {"role": "assistant", "content": "\n".join(blocks)}

    corruptions = {
        "remove_tool_call": remove_tool_call,
        "remove_required_argument": remove_required_argument,
        "corrupt_argument": corrupt_argument,
        "corrupt_json": corrupt_json,
    }
    # Sampling weights for picking which corruption to apply.
    corruption_weights = {
        "remove_tool_call": 0.3,
        "remove_required_argument": 0.3,
        "corrupt_argument": 0.3,
        "corrupt_json": 0.1,
    }

    def weighted_order():
        """Random try-order following corruption_weights (without replacement),
        so the first applicable corruption respects the target distribution."""
        remaining = list(corruptions)
        order = []
        while remaining:
            weights = [corruption_weights[name] for name in remaining]
            pick = random.choices(remaining, weights=weights, k=1)[0]
            order.append(pick)
            remaining.remove(pick)
        return order

    for doc in data:
        chosen_answer = doc.metadata["messages"][-1]
        # We can only corrupt an assistant turn that actually calls tools.
        if chosen_answer.get("role") != "assistant" or not chosen_answer.get(
            "tool_calls"
        ):
            continue

        tools_raw = doc.metadata.get("tools") or []
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw) if tools_raw else []
        required_map = _required_map(tools_raw)

        # Try the corruptions in a weighted-random order and keep the first
        # applicable one; corrupt_json applies to any tool-calling turn, so we
        # virtually always select a corruption.
        order = weighted_order()
        rejected_answer = None
        applied = None
        for name in order:
            success, rejected_answer = corruptions[name](chosen_answer, required_map)
            if success:
                applied = name
                break
        if applied is None:
            continue

        # Build the chosen/rejected conversations (shared context + the
        # respective answer) and flatten them to the NeMo-RL format. The
        # rejected answer may be None (e.g. corrupt_argument is filled in by a
        # later LLM step); in that case only the chosen side is written here.
        context = doc.metadata["messages"][:-1]
        doc.metadata["chosen"] = nemo_rl_format_messages(context + [chosen_answer])
        if rejected_answer is not None:
            doc.metadata["rejected"] = nemo_rl_format_messages(
                context + [rejected_answer]
            )
        doc.metadata["corruption"] = applied

        # Keep only the DPO fields in the final output. "tools" is never needed
        # downstream; "messages" is still needed by the corrupt_argument LLM
        # stage, so it is kept only for that (non-final) partition.
        doc.metadata.pop("tools", None)
        if applied != "corrupt_argument":
            doc.metadata.pop("messages", None)
        yield doc


# ---------------------------------------------------------------------------
# corrupt_argument stage: deferred to a small LLM.
#
# The corruption split above routes the "corrupt_argument" examples to their own
# partition with the chosen answer set but no rejected answer. This stage reads
# that partition and asks a small Qwen model to corrupt the arguments of a single
# tool call, then builds the rejected answer from the model's output.
# ---------------------------------------------------------------------------

CORRUPT_ARGUMENT_PROMPT = (
    "You are generating a HARD NEGATIVE example to train a tool-calling model.\n"
    "You are given a single tool call as a JSON object with a \"name\" and an "
    "\"arguments\" object.\n"
    "Produce a corrupted version of this tool call in which one or more ARGUMENT "
    "VALUES are wrong, so the call no longer answers the request correctly.\n"
    "Rules:\n"
    "- Keep the function \"name\" exactly the same.\n"
    "- Keep exactly the same argument keys (never add or drop a key).\n"
    "- Change one or more argument VALUES to plausible but INCORRECT values "
    "(e.g. a wrong city, a wrong number, swapped units, a wrong date). Keep each "
    "value's type and stay valid JSON.\n"
    "- Output ONLY the corrupted JSON object, nothing else."
)


def _tool_call_payload(tool_call):
    """Return the {name, arguments} dict of a tool call (flat or nested)."""
    src = tool_call.get("function", tool_call)
    return {"name": src["name"], "arguments": src.get("arguments", {}) or {}}


def _set_tool_call_arguments(tool_call, arguments):
    """Set the arguments of a tool call in place (flat or nested format)."""
    dst = tool_call["function"] if "function" in tool_call else tool_call
    dst["arguments"] = arguments


def _parse_json_object(text):
    """Best-effort extraction of a single JSON object from an LLM completion."""
    import json
    import re

    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def select_tool_call_to_corrupt(data, rank: int = 0, world_size: int = 1):
    """Pick, for each doc, one tool call that has arguments; its index is stored
    so the LLM is only shown that single call."""
    import random

    for doc in data:
        tool_calls = doc.metadata["messages"][-1].get("tool_calls") or []
        candidates = [
            i
            for i, tool_call in enumerate(tool_calls)
            if (tool_call.get("function", tool_call).get("arguments") or {})
        ]
        if not candidates:
            continue
        doc.metadata["target_call_idx"] = random.choice(candidates)
        yield doc


def corrupt_argument_query_builder(runner, doc):
    """Build the inference request: only the single tool call to corrupt is
    given to the model."""
    import json

    tool_call = doc.metadata["messages"][-1]["tool_calls"][
        doc.metadata["target_call_idx"]
    ]
    return {
        "messages": [
            {"role": "system", "content": CORRUPT_ARGUMENT_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_tool_call_payload(tool_call), ensure_ascii=False),
            },
        ],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
    }


def build_corrupted_rejected(self, doc):
    """Turn the LLM's corrupted tool call into the rejected answer.

    Keeps the original function name and argument keys; only the values coming
    from the model are used, and only when they actually differ. Docs where the
    model failed to produce valid, different arguments are written without a
    "rejected" key (the raw output is kept for inspection)."""
    import copy

    results = doc.metadata.pop("inference_results", None)
    # Inference results are InferenceSuccess objects here (only serialized to
    # dicts once written to jsonl), so read the text via attribute access.
    result = results[0] if results else None
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    doc.metadata["corrupt_argument_raw"] = text

    idx = doc.metadata["target_call_idx"]
    chosen_answer = doc.metadata["messages"][-1]
    original_args = _tool_call_payload(chosen_answer["tool_calls"][idx])["arguments"]

    parsed = _parse_json_object(text)
    new_args = parsed.get("arguments") if isinstance(parsed, dict) else None
    if isinstance(new_args, dict) and new_args != original_args:
        rejected_answer = copy.deepcopy(chosen_answer)
        _set_tool_call_arguments(rejected_answer["tool_calls"][idx], new_args)
        context = doc.metadata["messages"][:-1]
        doc.metadata["rejected"] = nemo_rl_format_messages(context + [rejected_answer])
        doc.metadata["state"] = "success"
    else:
        doc.metadata["state"] = "fail"

    # Drop the intermediate fields so the final output only carries the DPO
    # columns (chosen/rejected/corruption/...).
    for key in ("messages", "tools", "target_call_idx"):
        doc.metadata.pop(key, None)
    return doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tp", type=int, default=1)
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/xlam_oaiformat/data",
        ),
        create_rejected,
        JsonlWriter(
            f"{DATA_PATH}/xlam_dpo_oaiformat/data",
            output_filename="${corruption}/${rank}.jsonl",
            expand_metadata=True,
        ),
    ]

    create_rejected_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/xlam_dpo_oaiformat/logs",
        job_name="xlam_dpo_oaiformat",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )

    #########
    # corrupt_argument: corrupt the arguments with a small LLM. Depends on the
    # split above so it runs once the corrupt_argument partition is written.
    #########
    corrupt_argument_config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    corrupt_argument_pipeline = [
        JsonlReader(
            f"{DATA_PATH}/xlam_dpo_oaiformat/data/corrupt_argument",
            adapter=instruct_adapter,
        ),
        select_tool_call_to_corrupt,
        InferenceRunner(
            query_builder=corrupt_argument_query_builder,
            config=corrupt_argument_config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/xlam_dpo_oaiformat/corrupt_argument_llm/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/xlam_dpo_oaiformat/corrupt_argument_llm/data",
                output_filename="${state}/${rank}_chunk_${chunk_index}.jsonl",
                expand_metadata=True,
            ),
            postprocess_fn=build_corrupted_rejected,
            skip_bad_requests=True,
        ),
    ]

    corrupt_argument_executor = create_executor(
        corrupt_argument_pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/xlam_dpo_oaiformat/corrupt_argument_llm/logs",
        job_name="xlam_corrupt_argument_llm",
        tasks=1,
        time="01:00:00",
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
        depends=create_rejected_executor,
    )
    corrupt_argument_executor.run()
