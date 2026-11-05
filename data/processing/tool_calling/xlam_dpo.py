"""DPO pairs on xlam_oaiformat with the chosen and rejected answers sampled from
two different models, both checked against the ground truth (see
Verification). Built like posttraining/create_preference_data.py: one sampling
stage per model, then the pairing.

Stage 1 (GPU): the rejected model (e.g. a small model, or the model to train)
    samples ``--rejected_n_samples`` answers per prompt: the xlam_sft
    prompt (user query + ``tools``), after the seeded remove_tool perturbation.
Stage 2 (GPU): the rejected answers are verified and only the prompts with at
    least one usable failure (see is_rejectable) are kept; the chosen model
    (e.g. a stronger model) samples ``--chosen_n_samples`` answers on those
    prompts only.
Stage 3 (CPU): the chosen answers are verified. One pair per distinct usable
    failure of the rejected model (see build_pairs), against a correct answer
    of the chosen model drawn at random. Prompts where the chosen model has no
    correct answer give no pair.

Both models run in the same mode (``--enable_thinking`` or not), with the
sampling settings of generation_config. Chosen tool calls are rendered from the
ground truth and rejected ones from the parsed calls, so both sides use the
same tool-call layout whatever the model wrote (JSON or XML). The answer
lengths are recorded, to check for a length or style bias between the two
models.

Perturbation (``--remove_tool_ratio``, default 10% of the prompts): one tool
the ground truth calls is removed from the tools, and its calls from the
ground truth; if no call is left, the expected answer is a text (see
remove_tool).

Verification: parse each completion and compare its tool calls with the ground
    truth (same check as xlam_sft: same calls, order-insensitive, generated
    arguments cast with the schema types). Each answer gets ``correct`` and, when
    it fails, a ``failure`` label and a human-readable ``error``:
      - request_error: the request failed (no text returned)
      - not_finished: finish_reason != "stop" (e.g. hit max_tokens)
      - text_after_call: complete tool calls followed by text (the content
        must come before the calls)
      - unparsable: unclosed think, invalid tool call JSON/XML (BFCL
        ast_decoder:decoder_failed). Text before the calls is fine: only the
        calls are checked.
      - irrelevance_error: a tool call while a text answer is expected
      - relevance_error: a text answer while tool calls are expected
      - otherwise a BFCL AST checker error type, see classify_tool_calls.
    ``near_miss`` flags failures that BFCL would accept (string case or
    punctuation, optional parameters set to their default): likely not real
    mistakes, to keep out of rejected.
"""


import json
import re
from functools import partial

from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter

from utils import (
    create_parser,
    parse_args,
    create_executor,
    instruct_adapter,
    nemo_rl_format_messages,
)
from xlam_sft import (
    _DATA_DIR,
    TOOL_DATA_REVISION,
    resolve_input_path,
    _result_text,
    prepare,
    parse_generation,
    is_text_after_calls,
    same_tool_calls,
    _param_types,
    _canonical,
    _convert_value,
    _normalize_numbers,
    remove_tool,
    parse_text_answer,
    generation_config,
)

async def query_builder(runner, doc, gen_config: dict, n_samples: int = 5):
    """Yield ``n_samples`` identical payloads; sampling makes them differ.
    (The runner only reads choices[0], so vLLM's ``n`` cannot be used.)"""
    tools = doc.metadata["tool_schemas"]
    for _ in range(n_samples):
        payload = {"messages": doc.metadata["context"], **gen_config}
        if tools:  # no tools left after remove_tool: plain chat request
            payload["tools"] = tools
            # "none": tools stay in the prompt, but vLLM returns the raw text
            # instead of requiring --enable-auto-tool-choice + a tool parser.
            payload["tool_choice"] = "none"
        yield payload


#########
# Failure taxonomy, after the BFCL AST checker
# (gorilla/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py).
# The reference is the ground truth call (BFCL's possible_answer), not the
# schema: xlam's ``required`` lists are unreliable (most tools mark every
# parameter required, defaults included, and the ground truth omits some). The
# schema only tells a required parameter (in ``required`` and no ``default``)
# from an optional one. Labels are BFCL error types without the checker prefix:
#   wrong_count            different number of calls
#   wrong_func_name        no generated call has the expected function name
#   wrong_output_format    arguments are not an object (ast_decoder:decoder_wrong_output_format)
#   missing_required       a required ground-truth parameter is omitted
#   unexpected_param       a parameter that is not in the schema (hallucinated name)
#   extra_param            a schema parameter the ground truth does not set
#                          (not BFCL: BFCL folds it into unexpected_param, but
#                          accepts it when its possible_answer lists the value)
#   type_error:simple      value of another JSON type (e.g. "3" vs 3; 3 and 3.0
#                          are equal, see xlam_sft._normalize_numbers)
#   type_error:nested      list items of another type
#   value_error:string     wrong string (exact match, BFCL is case/punctuation-insensitive)
#   value_error:list/tuple wrong list
#   value_error:dict_key   dict with other keys
#   value_error:dict_value dict with a wrong value
#   value_error:list_dict_count  list of dicts of another length
#   value_error:others     wrong number / bool / null
#   missing_optional       an optional parameter set by the ground truth is omitted
#   unclear                no difference found (should not happen)
# Checks follow BFCL's order (first error wins): name, missing required,
# unexpected/extra parameters, then type and value of each generated parameter,
# then missing optional parameters.
#########


def _dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_type(value):
    return "number" if _is_number(value) else type(value).__name__


def _dict_error(param, expected: dict, value: dict):
    if set(expected) != set(value):
        return "value_error:dict_key", (
            f"Dict keys of {param!r}: {sorted(value)}, expected {sorted(expected)}."
        )
    for key in expected:
        if _dump(expected[key]) != _dump(value[key]):
            return "value_error:dict_value", (
                f"Invalid value for {param!r}[{key!r}]: {value[key]!r}, expected {expected[key]!r}."
            )
    return None


def _value_error(param, expected, value):
    """(label, message) for a parameter value that differs from the ground
    truth, or None if they are equal."""
    if _dump(expected) == _dump(value):
        return None
    if _is_number(expected) and _is_number(value):
        return "value_error:others", f"Invalid value for {param!r}: {value!r}, expected {expected!r}."
    if type(expected) is not type(value):
        return "type_error:simple", (
            f"Incorrect type for {param!r}. Expected {type(expected).__name__}, "
            f"got {type(value).__name__}. Parameter value: {value!r}."
        )
    if isinstance(expected, str):
        return "value_error:string", f"Invalid value for {param!r}: {value!r}, expected {expected!r}."
    if isinstance(expected, dict):
        return _dict_error(param, expected, value) or (
            "unclear", f"Dicts differ for {param!r}."
        )
    if isinstance(expected, list):
        if expected and all(isinstance(e, dict) for e in expected):
            if len(expected) != len(value):
                return "value_error:list_dict_count", (
                    f"{len(value)} dicts in {param!r}, expected {len(expected)}."
                )
            for e, v in zip(expected, value):
                if not isinstance(v, dict):
                    return "type_error:nested", f"Non-dict item in {param!r}: {v!r}."
                error = _dict_error(param, e, v)
                if error:
                    return error
        item_types = {_json_type(e) for e in expected}
        if item_types:
            for v in value:
                if _json_type(v) not in item_types:
                    return "type_error:nested", (
                        f"Incorrect item type in {param!r}: {v!r}, expected {sorted(item_types)}."
                    )
        return "value_error:list/tuple", f"Invalid value for {param!r}: {value!r}, expected {expected!r}."
    return "value_error:others", f"Invalid value for {param!r}: {value!r}, expected {expected!r}."


def _schemas(tools) -> dict:
    """{function name: parameters schema} from the tool schemas."""
    if isinstance(tools, str):
        tools = json.loads(tools)
    schemas = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        schemas[function["name"]] = function.get("parameters") or {}
    return schemas


def classify_call(generated: dict, expected: dict, schema: dict):
    """BFCL simple_function_checker, against one ground-truth call:
    (label, message) of the first error found."""
    if generated["name"] != expected["name"]:
        return "wrong_func_name", (
            f"Function name {expected['name']!r} expected, got {generated['name']!r}."
        )
    arguments = generated["arguments"]
    if not isinstance(arguments, dict):
        return "wrong_output_format", f"Arguments are not an object: {arguments!r}."
    reference = expected["arguments"]
    properties = schema.get("properties") or {}
    required = {
        p for p in schema.get("required") or []
        if "default" not in properties.get(p, {})
    }
    missing = [p for p in reference if p not in arguments]
    for param in missing:
        if param in required:
            return "missing_required", f"Missing required parameter: {param!r}."
    for param in arguments:
        if param not in properties:
            return "unexpected_param", f"Unexpected parameter (not in the schema): {param!r}."
    for param in arguments:
        if param not in reference:
            return "extra_param", (
                f"Parameter not set by the ground truth: {param!r}={arguments[param]!r}."
            )
    for param, value in arguments.items():
        error = _value_error(param, reference[param], value)
        if error:
            return error
    if missing:
        return "missing_optional", f"Optional parameter {missing[0]!r} not provided."
    return "unclear", "No difference found."


def classify_tool_calls(generated, ground_truth, tools):
    """(label, message) explaining why ``generated`` differs from
    ``ground_truth`` (BFCL parallel_function_checker_no_order). Calls equal to
    a ground-truth call are matched first; the first unmatched ground-truth
    call is then compared with a leftover generated call of the same name
    (any leftover if none has it)."""
    if len(generated) != len(ground_truth):
        return "wrong_count", f"{len(generated)} calls, expected {len(ground_truth)}."
    param_types = _param_types(tools)
    generated = [json.loads(_canonical(c, param_types)) for c in generated]
    expected = [json.loads(_canonical(c)) for c in ground_truth]
    leftover = list(generated)
    unmatched = []
    for call in expected:
        match = next((g for g in leftover if _dump(g) == _dump(call)), None)
        if match is None:
            unmatched.append(call)
        else:
            leftover.remove(match)
    if not unmatched:
        return "unclear", "No difference found."
    call = unmatched[0]
    candidate = next((g for g in leftover if g["name"] == call["name"]), leftover[0])
    return classify_call(candidate, call, _schemas(tools).get(call["name"], {}))


def standardize_string(input_string: str):
    """BFCL's string normalization (ast_checker.standardize_string): drop
    spaces and ",./-_*^", lowercase, single to double quotes."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", input_string).lower().replace("'", '"')


def _lenient_equal(expected, value) -> bool:
    """Equality as BFCL checks values: strings compared after
    standardize_string, dict keys compared exactly. (Numbers already compare
    by value: both sides go through _canonical.)"""
    if isinstance(expected, str) and isinstance(value, str):
        return standardize_string(expected) == standardize_string(value)
    if isinstance(expected, list) and isinstance(value, list):
        return len(expected) == len(value) and all(map(_lenient_equal, expected, value))
    if isinstance(expected, dict) and isinstance(value, dict):
        return set(expected) == set(value) and all(
            _lenient_equal(expected[k], value[k]) for k in expected
        )
    return _dump(expected) == _dump(value)


_DESCRIPTION_DEFAULT = re.compile(
    r"\bdefaults?(?:\s+value)?\s*(?:to|is|:|=)\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s,;()]+))"
    r"|(?:'([^']*)'|\"([^\"]*)\")\s*\(default\)",
    re.IGNORECASE,
)


def _description_default(description: str):
    """Default written in a parameter description ("Defaults to 1.",
    "Default is 'en'.", '"copper" (default)'), as text ("" for "Defaults to
    None"), or None if there is none. xlam often gives defaults this way only,
    without a schema default."""
    match = _DESCRIPTION_DEFAULT.search(description or "")
    if not match:
        return None
    quoted = next((g for g in match.groups()[:2] + match.groups()[3:] if g is not None), None)
    value = quoted if quoted is not None else match.group(3).rstrip(".")
    return "" if value.lower() in ("none", "null") else value


def _schema_defaults(tools) -> dict:
    """{function name: {parameter: default value}} for the parameters whose
    schema has a default, or else whose description gives one, cast with the
    declared type (xlam often stores defaults as text, e.g. "166" for an int).
    A None or empty default is "": the model setting "" leaves it unset, as
    BFCL accepts "" for an optional parameter."""
    if isinstance(tools, str):
        tools = json.loads(tools)
    defaults = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        properties = (function.get("parameters") or {}).get("properties") or {}
        function_defaults = {}
        for name, prop in properties.items():
            default = prop.get("default")
            if default in (None, ""):
                described = _description_default(prop.get("description"))
                if described is None and "default" not in prop:
                    continue
                default = described or ""
            function_defaults[name] = (
                default if default == "" else _normalize_numbers(_convert_value(default, str(prop.get("type", ""))))
            )
        defaults[function["name"]] = function_defaults
    return defaults


def is_near_miss(generated, ground_truth, tools) -> bool:
    """True if the calls only differ by what BFCL accepts: strings that differ
    in case, spaces or punctuation, and optional parameters the ground truth
    does not set but the model sets to their default (from the schema or the
    description, see _schema_defaults; BFCL's reference
    answers list the default as an accepted value, whether or not the schema
    marks the parameter required). Such a failure is likely not a real mistake."""
    if len(generated) != len(ground_truth):
        return False
    param_types = _param_types(tools)
    defaults = _schema_defaults(tools)

    def matches(call, generated_call):
        arguments = generated_call["arguments"]
        if generated_call["name"] != call["name"] or not isinstance(arguments, dict):
            return False
        function_defaults = defaults.get(call["name"], {})
        arguments = {
            k: v for k, v in arguments.items()
            if k in call["arguments"]
            or not (k in function_defaults and _lenient_equal(function_defaults[k], v))
        }
        return _lenient_equal(call["arguments"], arguments)

    leftover = [json.loads(_canonical(c, param_types)) for c in generated]
    for call in (json.loads(_canonical(c)) for c in ground_truth):
        match = next((g for g in leftover if matches(call, g)), None)
        if match is None:
            return False
        leftover.remove(match)
    return True


def verify_sample(sample, ground_truth, tools, enable_thinking: bool = True):
    """Annotate one sample in place with ``correct``, ``failure`` / ``error``
    (None when correct), ``near_miss`` (see is_near_miss; False unless the
    calls differ), and the parsed ``reasoning`` (None without thinking) /
    ``tool_calls`` / ``content`` (text answer, or text before the calls: only
    the calls are checked) when parsable.

    An empty ``ground_truth`` (all its tools removed, see remove_tool) expects
    a text answer: any tool call is an ``irrelevance_error`` (BFCL's label for
    calling a function when none fits). The other way round, a text answer
    when calls are expected is a ``relevance_error``."""
    sample["reasoning"] = None
    sample["tool_calls"] = None
    sample["content"] = None
    sample["near_miss"] = False
    error = None
    if sample.get("generation") is None:
        failure = "request_error"
    elif sample.get("finish_reason") != "stop":
        failure = "not_finished"
    else:
        text = sample["generation"]
        if not enable_thinking:
            # The template puts the empty think block in the prompt, so the
            # completion is just the answer: close a (empty) reasoning for
            # parse_generation. A <think> written anyway is a second
            # reasoning block, hence unparsable.
            text = "</think>" + text
        parsed = parse_generation(text)
        if parsed is not None:
            reasoning, sample["content"], calls = parsed
            parsed = reasoning, calls
        text_answer = parse_text_answer(text) if parsed is None else None
        if parsed is None and is_text_after_calls(text):
            failure = "text_after_call"
            error = "Text after the tool calls (content must come before them)."
        elif not ground_truth:
            if parsed is not None:
                reasoning, sample["tool_calls"] = parsed
                sample["reasoning"] = reasoning if enable_thinking else None
                failure = "irrelevance_error"
                error = "Called {} while no available tool fits.".format(
                    sorted({c.get("function", c)["name"] for c in sample["tool_calls"]})
                )
            elif text_answer is not None:
                reasoning, sample["content"] = text_answer
                sample["reasoning"] = reasoning if enable_thinking else None
                failure = None
            else:
                failure = "unparsable"
        elif parsed is None and text_answer is not None:
            # BFCL's relevance_error: no call although a tool fits.
            reasoning, sample["content"] = text_answer
            sample["reasoning"] = reasoning if enable_thinking else None
            failure = "relevance_error"
            error = "No tool call, expected {}.".format(
                sorted({c.get("function", c)["name"] for c in ground_truth})
            )
        elif parsed is None:
            failure = "unparsable"
        else:
            reasoning, sample["tool_calls"] = parsed
            sample["reasoning"] = reasoning if enable_thinking else None
            if same_tool_calls(sample["tool_calls"], ground_truth, tools):
                failure = None
            else:
                failure, error = classify_tool_calls(
                    sample["tool_calls"], ground_truth, tools
                )
                sample["near_miss"] = is_near_miss(
                    sample["tool_calls"], ground_truth, tools
                )
    sample["correct"] = failure is None
    sample["failure"] = failure
    sample["error"] = error
    return sample


# Failed samples that are not used as rejected: no text, truncated (no complete
# answer to prefer against), or likely not a real mistake (see is_near_miss).
UNUSABLE_FAILURES = ("request_error", "not_finished")


def is_rejectable(sample) -> bool:
    return (
        not sample["correct"]
        and sample["failure"] not in UNUSABLE_FAILURES
        and not sample["near_miss"]
    )


def _assistant_message(sample, tool_calls, enable_thinking: bool) -> dict:
    """Nested assistant message of a sample; NemoRLFormat-style flattening is
    done by the caller. Text answers (expected text, see remove_tool) keep
    their reasoning and text; unparsable samples keep their raw text."""
    if tool_calls is None and sample.get("content") is not None:
        message = {"role": "assistant", "content": sample["content"]}
    elif tool_calls is None:
        text = sample["generation"]
        return {"role": "assistant", "content": text if enable_thinking else text.strip()}
    else:
        # Text written before the calls is kept as the content.
        message = {"role": "assistant", "content": sample.get("content") or "", "tool_calls": tool_calls}
    if enable_thinking:
        message["reasoning_content"] = sample["reasoning"]
    return message


def _rejected_key(sample, calls) -> str:
    """What a rejected answer says, without its reasoning: its text and tool
    calls (order-insensitive), or the raw answer when unparsable."""
    if calls is None:
        return (sample["generation"] or "").split("</think>", 1)[-1].strip()
    return json.dumps(
        {"content": sample.get("content") or "", "calls": sorted(_dump(c) for c in calls)},
        sort_keys=True,
    )


SIDES = ("rejected", "chosen")


def _short(model: str) -> str:
    """Folder-friendly model name: Qwen/Qwen3-0.6B -> qwen3-0.6b."""
    return model.rstrip("/").split("/")[-1].lower()


def postprocess(self, doc, side: str, gen_config_meta: dict):
    """Store the sampled answers of one side in ``{side}_samples`` and its
    request settings in ``generation_config_{side}``."""
    results = doc.metadata.pop("inference_results", None) or []
    doc.metadata[f"{side}_samples"] = [
        {
            "generation": _result_text(res),
            "finish_reason": getattr(res, "finish_reason", None),
        }
        for res in results
    ]
    doc.metadata[f"generation_config_{side}"] = gen_config_meta
    return doc


def verify_side(data, rank: int = 0, world_size: int = 1, side: str = "rejected"):
    """xlam_dpo.verify_samples for one side: annotate ``{side}_samples`` and
    set ``{side}_n_correct``. The tool schemas are kept (the chosen stage
    still needs them)."""
    for doc in data:
        meta = doc.metadata
        enable_thinking = meta[f"generation_config_{side}"]["chat_template_kwargs"]["enable_thinking"]
        for sample in meta[f"{side}_samples"]:
            verify_sample(sample, meta["ground_truth"], meta.get("tools"), enable_thinking)
        meta[f"{side}_n_correct"] = sum(s["correct"] for s in meta[f"{side}_samples"])
        yield doc


def keep_with_failures(data, rank: int = 0, world_size: int = 1):
    """Only prompts where the rejected model has a usable failure need chosen
    answers."""
    for doc in data:
        if any(is_rejectable(s) for s in doc.metadata["rejected_samples"]):
            yield doc


def build_pairs(data, rank: int = 0, world_size: int = 1):
    """xlam_dpo.build_pairs with the rejected answers from the rejected model
    and the chosen ones from the chosen model: one pair per distinct usable
    failure, against a random correct chosen answer (seeded with the prompt
    id)."""
    import random

    from datatrove.data import Document

    for doc in data:
        meta = doc.metadata
        meta.pop("tool_schemas", None)
        chosen_samples, rejected_samples = meta["chosen_samples"], meta["rejected_samples"]
        correct = [i for i, s in enumerate(chosen_samples) if s["correct"]]
        rejected = [i for i, s in enumerate(rejected_samples) if is_rejectable(s)]
        if not correct or not rejected:
            continue
        thinking = {
            side: meta[f"generation_config_{side}"]["chat_template_kwargs"]["enable_thinking"]
            for side in SIDES
        }
        param_types = _param_types(meta.get("tools"))
        original = meta["messages_original"]
        if isinstance(original, str):
            original = json.loads(original)
        context = original[:-1]
        rng = random.Random(str(doc.id))
        cast_calls = {
            i: None if rejected_samples[i]["tool_calls"] is None
            else [json.loads(_canonical(c, param_types)) for c in rejected_samples[i]["tool_calls"]]
            for i in rejected
        }
        groups = {}  # distinct answer -> indices, in order of first appearance
        for i in rejected:
            groups.setdefault(_rejected_key(rejected_samples[i], cast_calls[i]), []).append(i)
        for group in groups.values():
            rejected_index = rng.choice(group)
            chosen_index = rng.choice(correct)
            bad = rejected_samples[rejected_index]
            chosen = nemo_rl_format_messages(context + [_assistant_message(
                chosen_samples[chosen_index], meta["ground_truth"] or None, thinking["chosen"]
            )])
            rejected_messages = nemo_rl_format_messages(context + [_assistant_message(
                bad, cast_calls[rejected_index], thinking["rejected"]
            )])
            yield Document(
                text="",
                id=f"{doc.id}_{rejected_index}",
                metadata={
                    "dataset": meta.get("dataset") or "",
                    "tools": meta["tools"],
                    "chosen": chosen,
                    "rejected": rejected_messages,
                    "chosen_model": meta["generation_config_chosen"]["model_name_or_path"],
                    "rejected_model": meta["generation_config_rejected"]["model_name_or_path"],
                    "chosen_index": chosen_index,
                    "rejected_index": rejected_index,
                    "rejected_failure": bad["failure"],
                    "rejected_error": bad["error"],
                    "removed_tool": meta.get("removed_tool"),
                    # characters of the assistant answers, to look for a length bias
                    "chosen_length": len(chosen[-1]["content"]),
                    "rejected_length": len(rejected_messages[-1]["content"]),
                    "generation_config_chosen": meta["generation_config_chosen"],
                    "generation_config_rejected": meta["generation_config_rejected"],
                },
            )


def inference_config(model, tp, temperature, args):
    return InferenceConfig(
        server_type="vllm",
        model_name_or_path=model,
        tp=tp,
        temperature=temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=args.max_concurrent_requests,
        max_concurrent_tasks=args.max_concurrent_requests,
        metric_interval=120,
        model_kwargs={
            "max-num-seqs": args.max_concurrent_requests,
            **({"gpu-memory-utilization": args.gpu_memory_utilization}
               if args.gpu_memory_utilization else {}),
        },
    )


def gpu_executor(pipeline, name, tp, args, depends=None):
    return create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.data_path}/{name}/logs",
        job_name=name.replace("/", "_"),
        tasks=1,
        time="20:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": f"gpu:{tp}",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
        depends=depends,
    )


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--chosen_model", type=str, default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--rejected_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--chosen_tp", type=int, default=1)
    parser.add_argument("--rejected_tp", type=int, default=1)
    parser.add_argument("--chosen_n_samples", type=int, default=1,
                        help="Answers per prompt from the chosen model (one correct is enough)")
    parser.add_argument("--rejected_n_samples", type=int, default=5,
                        help="Answers per prompt from the rejected model")
    parser.add_argument("--rejected_input", type=str, default=None,
                        help="Folder of rejected-model answers to use instead of stage 1's output "
                             "(stage 1 is then neither run nor waited for), e.g. a batch copied "
                             "from another machine while stage 1 is still running there")
    parser.add_argument("--chosen_batch", type=str, default="all",
                        help="Name of this chosen-stage batch: its answers go to chosen/<name>/, "
                             "and the pairs are built from all the batches")
    parser.add_argument("--stop_after", type=str, default=None, choices=["rejected", "chosen"],
                        help="Run only up to this stage (e.g. rejected on one machine, the rest "
                             "on another: completed stages are skipped when relaunched)")
    parser.add_argument("--revision", type=str, default=TOOL_DATA_REVISION,
                        help="Commit of OpenLLM-France/tool_data to read xlam_oaiformat from (must be in the HF cache)")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Override: xlam_oaiformat parquet dir (skips the cache lookup)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling (defaults: 0.6 thinking, 0.7 non-thinking)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking (both models).")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate (defaults: 4096 thinking, 2048 non-thinking)")
    parser.add_argument("--model_max_context", type=int, default=32768)
    parser.add_argument("--remove_tool_ratio", type=float, default=0.1,
                        help="Fraction of prompts where a tool called by the ground truth is "
                             "removed from the tools (its calls too; text answer if none left)")
    # Seed of the remove_tool draw (which prompts lose which tool), salted with
    # the prompt id. Default 1 here and 0 in xlam_sft, so the DPO and SFT data
    # perturb different prompts. The xlam_dpo data in tool_data (b9be666e) was
    # built with the draw of seed 0.
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed of the remove_tool draw (salted with the prompt id); use a "
                             "different seed per dataset (xlam_sft default: 0)")
    parser.add_argument("--max_concurrent_requests", type=int, default=500,
                        help="Requests in flight (also vLLM --max-num-seqs); raise it for small models")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="vLLM --gpu-memory-utilization (default: vLLM's 0.9); "
                             "lower it on unified-memory machines (DGX Spark)")
    args = parse_args(parser)
    DATA_PATH = args.data_path
    mode = "_thinking" if args.enable_thinking else ""
    # The rejected stage does not depend on the chosen model: shared by all the
    # chosen models paired with it.
    REJECTED_NAME = f"xlam_dpo_rejected_{_short(args.rejected_model)}{mode}"
    PAIRS_NAME = f"xlam_dpo_{_short(args.chosen_model)}__{_short(args.rejected_model)}{mode}"

    gen_config = generation_config(
        temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens
    )

    def gen_config_meta(model, n_samples):
        return {
            "model_name_or_path": model,
            **gen_config,
            "n_samples": n_samples,
            "remove_tool_ratio": args.remove_tool_ratio,
            "remove_tool_seed": args.seed,
            "input_dataset": "OpenLLM-France/tool_data",
            "input_revision": args.revision,
        }

    #########
    # Stage 1: rejected model, all prompts (GPU)
    #########
    rejected_executor = None
    if not args.rejected_input:  # stage 1 is not needed when its output is given
        input_path = resolve_input_path(args.revision, args.input_path)
        print(f"Reading xlam_oaiformat from {input_path}")
        rejected_executor = gpu_executor(
            [
                ParquetReader(input_path, adapter=instruct_adapter),
                prepare,
                partial(remove_tool, ratio=args.remove_tool_ratio, seed=args.seed),
                InferenceRunner(
                    query_builder=partial(query_builder, gen_config=gen_config, n_samples=args.rejected_n_samples),
                    config=inference_config(args.rejected_model, args.rejected_tp, gen_config["temperature"], args),
                    records_per_chunk=100,
                    checkpoints_local_dir=f"{DATA_PATH}/{REJECTED_NAME}/checkpoints",
                    output_writer=JsonlWriter(
                        f"{DATA_PATH}/{REJECTED_NAME}/generations",
                        output_filename="${rank}_chunk_${chunk_index}.jsonl",
                        expand_metadata=True,
                    ),
                    postprocess_fn=partial(
                        postprocess, side="rejected",
                        gen_config_meta=gen_config_meta(args.rejected_model, args.rejected_n_samples),
                    ),
                    skip_bad_requests=True,
                ),
            ],
            REJECTED_NAME, args.rejected_tp, args,
        )
    rejected_input = [JsonlReader(
        args.rejected_input or f"{DATA_PATH}/{REJECTED_NAME}/generations",
        glob_pattern="*.jsonl*", adapter=instruct_adapter,
    )]
    CHOSEN_DIR = f"{DATA_PATH}/{PAIRS_NAME}/chosen/{args.chosen_batch}"

    #########
    # Stage 2: chosen model, only on prompts with a usable rejected failure (GPU)
    #########
    chosen_executor = gpu_executor(
        rejected_input + [
            partial(verify_side, side="rejected"),
            keep_with_failures,
            InferenceRunner(
                query_builder=partial(query_builder, gen_config=gen_config, n_samples=args.chosen_n_samples),
                config=inference_config(args.chosen_model, args.chosen_tp, gen_config["temperature"], args),
                records_per_chunk=100,
                checkpoints_local_dir=f"{CHOSEN_DIR}/checkpoints",
                output_writer=JsonlWriter(
                    f"{CHOSEN_DIR}/generations",
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",
                    expand_metadata=True,
                ),
                postprocess_fn=partial(
                    postprocess, side="chosen",
                    gen_config_meta=gen_config_meta(args.chosen_model, args.chosen_n_samples),
                ),
                skip_bad_requests=True,
            ),
        ],
        f"{PAIRS_NAME}/chosen/{args.chosen_batch}", args.chosen_tp, args,
        depends=rejected_executor,
    )

    #########
    # Stage 3: verify the chosen answers and pair them (CPU)
    #########
    pairs_executor = create_executor(
        [
            # all the chosen batches so far
            JsonlReader(f"{DATA_PATH}/{PAIRS_NAME}/chosen", glob_pattern="*/generations/*.jsonl*",
                        adapter=instruct_adapter),
            partial(verify_side, side="chosen"),
            JsonlWriter(f"{DATA_PATH}/{PAIRS_NAME}/verified", expand_metadata=True),
            build_pairs,
            JsonlWriter(f"{DATA_PATH}/{PAIRS_NAME}/pairs", expand_metadata=True),
        ],
        local=args.local,
        debug=args.debug,
        # one logging dir per batch: the pairs are rebuilt after each batch
        logging_dir=f"{DATA_PATH}/{PAIRS_NAME}/logs_pairs/{args.chosen_batch}",
        job_name=f"{PAIRS_NAME}_pairs",
        tasks=1,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
        depends=chosen_executor,
    )
    {"rejected": rejected_executor, "chosen": chosen_executor, None: pairs_executor}[args.stop_after].run()
