import os
import json
import importlib.util

# Directly load posttraining/utils.py under a unique module name to avoid
# a circular import, since this file is also named 'utils'. It re-exports the
# pretraining helpers and defines the shared instruct/formatting utilities.

spec = importlib.util.spec_from_file_location(
    "posttraining_utils",
    os.path.join(os.path.dirname(__file__), "..", "posttraining", "utils.py"),
)
posttraining_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(posttraining_utils)

create_parser = posttraining_utils.create_parser
parse_args = posttraining_utils.parse_args
create_executor = posttraining_utils.create_executor
add_sampler_filter = posttraining_utils.add_sampler_filter
_custom_adapter_for_hf = posttraining_utils._custom_adapter_for_hf
HF_SCHEMA = posttraining_utils.HF_SCHEMA
FT176_LANGUAGES = posttraining_utils.pretraining_utils.FT176_LANGUAGES
instruct_adapter = posttraining_utils.instruct_adapter
FilterChinese = posttraining_utils.FilterChinese
apply_chat_template = posttraining_utils.apply_chat_template
format_tool_calls = posttraining_utils.format_tool_calls
nemo_rl_format_messages = posttraining_utils.nemo_rl_format_messages
NemoRLFormat = posttraining_utils.NemoRLFormat
check_last_message = posttraining_utils.check_last_message
TOKEN_COUNT_BUCKETS = posttraining_utils.TOKEN_COUNT_BUCKETS
token_count_label = posttraining_utils.token_count_label
ChatTokenCount = posttraining_utils.ChatTokenCount

def from_tools_to_system(system_content, tools, tokenizer):
    import re
    system_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}],
        tools=tools,
        tokenize=False,
    )
    system_prompt = re.search(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
    ).group(1)
    return system_prompt

def hub_adapter(self, document):
    """Row adapter for the Hub writers: emit only the shared columns.

    The default adapter writes every metadata key as its own column
    (``expand_metadata``), which gives each dataset a different schema -- so one
    ``load_dataset`` over the repo cannot union them -- and lets an optional
    field that is ``None`` in the first row type that column ``null``, after
    which the whole batch is rejected ("Table schema does not match schema used
    to create file"). Pinning the columns here gives every dataset in the repo
    one stable schema.

    ``dataset`` is the source id set by HuggingFaceDatasetReader; readers that
    do not set it (Parquet/Jsonl) leave it empty rather than null, so the column
    stays typed.

    ``messages_original`` is the JSON string of the messages before the NeMo-RL
    conversion (saved by ``NemoRLFormat`` in ``metadata["messages_original"]``);
    it is an empty string when the pipeline did not save it.

    ``language`` is the language label set by datatrove's ``LanguageFilter``
    (``metadata["language"]``); it is an empty string when the pipeline has no
    language annotation, so the column stays typed.
    """
    import json

    tools = document.metadata.get("tools")
    messages = document.metadata.get("messages") or []
    messages_original = document.metadata.get("messages_original") or ""
    if not isinstance(messages_original, str):
        messages_original = json.dumps(messages_original, ensure_ascii=False)
    return {
        "id": str(document.id),
        "dataset": str(document.metadata.get("dataset") or ""),
        "tools": tools if isinstance(tools, str) else json.dumps(tools or []),
        "messages": [
            {"role": str(m.get("role", "")), "content": str(m.get("content") or "")}
            for m in messages
        ],
        "messages_original": messages_original,
        "language": str(document.metadata.get("language") or ""),
    }


def hub_dpo_adapter(self, document):
    """Row adapter for the DPO Hub writers: emit only the five shared columns.

    Same rationale as ``hub_adapter`` (one stable schema across the repo), with
    the ``chosen`` / ``rejected`` conversations instead of ``messages``.
    """
    import json

    tools = document.metadata.get("tools")

    def format_messages(messages):
        return [
            {"role": str(m.get("role", "")), "content": str(m.get("content") or "")}
            for m in messages or []
        ]

    return {
        "id": str(document.id),
        "dataset": str(document.metadata.get("dataset") or ""),
        "tools": tools if isinstance(tools, str) else json.dumps(tools or []),
        "chosen": format_messages(document.metadata.get("chosen")),
        "rejected": format_messages(document.metadata.get("rejected")),
    }


def add_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
    system_key=None,
):
    import json
    for doc in data:
        tools = doc.metadata.get("tools", [])
        doc.metadata["tools"] = json.dumps(tools)

        messages = doc.metadata["messages"]
        if system_key is None:
            # No explicit source declared: use the leading system message (if
            # any) as the system content and drop it, so we rebuild it with the
            # tool schemas baked in instead of leaving two system messages.
            if messages and messages[0]["role"] == "system":
                system_content = messages[0]["content"]
                messages = messages[1:]
            else:
                system_content = ""
        else:
            system_content = doc.metadata.get(system_key, "")

        system_prompt = from_tools_to_system(system_content, tools, tokenizer)
        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + messages
        yield doc


class ToolTypeCheck(posttraining_utils.BaseFilter):
    """Check the JSON-Schema types of tool definitions and of tool-call arguments.

    Works before or after ``NemoRLFormat``: calls are read from ``message["tool_calls"]``
    (OpenAI format, ``arguments`` as a dict or a JSON string) or, when absent, parsed from
    the ``<tool_call>...</tool_call>`` blocks of the assistant content. Tools are read from
    ``metadata[tools_key]`` (a list or its JSON string).

    Issues (one ``stat_update`` per kind, counted once per document):
      def_invalid_type      a ``type`` that is not a JSON-Schema type (e.g. "float", "int", "str")
      def_bad_parameters    ``parameters`` is not a dict / null
      call_not_json         a <tool_call> block or its arguments do not parse as JSON
      call_args_not_object  ``arguments`` is not a JSON object
      call_unknown_tool     the called tool is not in the document's tools
      call_missing_required a required argument is missing
      call_extra_arg        an argument that is not a declared property
      call_type_mismatch    a value of the wrong type ("5" for a number, "[1, 2]" for an array, ...)
      call_enum_mismatch    a value outside the declared ``enum``

    Args:
        drop_on: issue kinds that drop the document (default: all). Others are only counted.
        allow_null_optional: accept ``null`` for an argument that is not required.
        allow_corrected: keep self-correction traces -- a bad call is ignored when a later
            call to the same tool in the conversation is valid (error -> retry rows).
        annotate_key: if set, the list of issues is stored in ``metadata[annotate_key]``.
        exclusion_writer: dropped documents (``filter_reason`` = first dropping issue kind).
    """

    name = "🔧 Tool Type Check"

    ISSUES = (
        "def_invalid_type",
        "def_bad_parameters",
        "call_not_json",
        "call_args_not_object",
        "call_unknown_tool",
        "call_missing_required",
        "call_extra_arg",
        "call_type_mismatch",
        "call_enum_mismatch",
    )
    VALID_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}
    _SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
    _SUBSCHEMA_DICTS = ("properties", "$defs", "definitions", "patternProperties")
    _SUBSCHEMA = ("items", "additionalProperties", "not", "contains")

    def __init__(
        self,
        messages_key: str = "messages",
        tools_key: str = "tools",
        drop_on=ISSUES,
        allow_null_optional: bool = True,
        allow_corrected: bool = True,
        annotate_key: str | None = None,
        exclusion_writer=None,
    ):
        super().__init__(exclusion_writer)
        unknown = set(drop_on) - set(self.ISSUES)
        if unknown:
            raise ValueError(f"Unknown issue kinds in drop_on: {unknown}")
        self.messages_key = messages_key
        self.tools_key = tools_key
        self.drop_on = set(drop_on)
        self.allow_null_optional = allow_null_optional
        self.allow_corrected = allow_corrected
        self.annotate_key = annotate_key

    # ---------------------------------------------------------------- schema helpers
    def _invalid_types(self, schema, path, out):
        if isinstance(schema, list):
            for i, sub in enumerate(schema):
                self._invalid_types(sub, f"{path}[{i}]", out)
            return
        if not isinstance(schema, dict):
            return
        t = schema.get("type")
        if t is not None:
            for x in t if isinstance(t, list) else [t]:
                if not isinstance(x, str) or x not in self.VALID_TYPES:
                    out.append(("def_invalid_type", path, repr(x)))
        for k, v in schema.items():
            if k in self._SUBSCHEMA_DICTS and isinstance(v, dict):
                for pk, pv in v.items():
                    self._invalid_types(pv, f"{path}.{pk}", out)
            elif k in self._SUBSCHEMA or (k in self._SUBSCHEMA_LISTS and isinstance(v, list)):
                self._invalid_types(v, f"{path}[{k}]", out)

    @staticmethod
    def _type_ok(value, t):
        if t == "integer":
            return (isinstance(value, int) and not isinstance(value, bool)) or (
                isinstance(value, float) and value.is_integer()
            )
        if t == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        py = {"string": str, "boolean": bool, "array": list, "object": dict, "null": type(None)}
        return isinstance(value, py[t])

    def _value_issues(self, value, schema, path, out):
        """Type / enum check of a value against a (sub)schema; recurses into objects and arrays."""
        if not isinstance(schema, dict):
            return
        for key in ("anyOf", "oneOf"):
            if isinstance(schema.get(key), list):
                if not any(self._value_ok(value, sub) for sub in schema[key]):
                    out.append(("call_type_mismatch", path, f"{key}: {type(value).__name__}"))
                return
        t = schema.get("type")
        types = [x for x in (t if isinstance(t, list) else [t]) if isinstance(x, str) and x in self.VALID_TYPES]
        if types and not any(self._type_ok(value, x) for x in types):
            out.append(("call_type_mismatch", path, f"expected {types}, got {type(value).__name__}"))
            return
        if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
            out.append(("call_enum_mismatch", path, repr(value)[:80]))
        if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
            for k, v in value.items():
                if k in schema["properties"]:
                    self._value_issues(v, schema["properties"][k], f"{path}.{k}", out)
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for v in value:
                self._value_issues(v, schema["items"], f"{path}[]", out)

    def _value_ok(self, value, schema):
        out = []
        self._value_issues(value, schema, "", out)
        return not out

    @staticmethod
    def _declared(params):
        """(properties, required) including the alternatives of a top-level anyOf/oneOf/allOf."""
        props = dict(params.get("properties") or {})
        required = set(params.get("required") or [])
        for k in ("anyOf", "oneOf", "allOf"):
            for sub in params.get(k) or []:
                if isinstance(sub, dict):
                    props.update(sub.get("properties") or {})
        return props, required

    # ---------------------------------------------------------------- call helpers
    @staticmethod
    def _iter_calls(message):
        """Yield (name, arguments) per tool call; arguments is None if it is not valid JSON."""
        import ast
        import re

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, str):  # some sources store the repr / JSON string of the list
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                try:
                    tool_calls = ast.literal_eval(tool_calls)
                except (ValueError, SyntaxError):
                    yield None, None
                    return
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = None
                yield fn.get("name"), args
            return
        for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", message.get("content") or "", re.S):
            try:
                call = json.loads(raw)
            except json.JSONDecodeError:
                yield None, None
                continue
            args = call.get("arguments", {}) if isinstance(call, dict) else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            yield (call.get("name") if isinstance(call, dict) else None), args

    def _call_issues(self, name, args, tools):
        if name is None and args is None:
            return [("call_not_json", None, "")]
        if args is None:
            return [("call_not_json", name, "arguments")]
        if not isinstance(args, dict):
            return [("call_args_not_object", name, type(args).__name__)]
        if name not in tools:
            return [("call_unknown_tool", name, "")]
        params = tools[name]
        props, required = self._declared(params)
        out = [("call_missing_required", f"{name}.{k}", "") for k in required if k not in args]
        # Extra args are only checked when the tool declares its properties (an explicit
        # ``"properties": {}`` means "no argument"); ``parameters: null`` / no properties accepts any.
        declared = bool(props) or "properties" in params or params.get("additionalProperties") is False
        out += [("call_extra_arg", f"{name}.{k}", "") for k in args if declared and k not in props]
        for k, v in args.items():
            if k not in props:
                continue
            if v is None and self.allow_null_optional and k not in required:
                continue
            self._value_issues(v, props[k], f"{name}.{k}", out)
        return out

    # ---------------------------------------------------------------- filter
    def filter(self, doc):
        tools_raw = doc.metadata.get(self.tools_key) or []
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw) if tools_raw.strip() else []

        issues = []
        tools = {}
        for tool in tools_raw:
            fn = tool.get("function", tool)
            params = fn.get("parameters")
            if params is None:
                params = {}
            elif not isinstance(params, dict):
                issues.append(("def_bad_parameters", fn.get("name"), type(params).__name__))
                params = {}
            self._invalid_types(params, fn.get("name"), issues)
            tools[fn.get("name")] = params

        # Per call: (tool name, issues). A bad call followed by a valid call to the same tool is
        # a self-correction trace and is not reported when allow_corrected is set.
        calls = []
        for message in doc.metadata.get(self.messages_key) or []:
            if message.get("role") == "assistant":
                for name, args in self._iter_calls(message):
                    calls.append((name, self._call_issues(name, args, tools)))
        for i, (name, call_issues) in enumerate(calls):
            if call_issues and self.allow_corrected and name is not None and any(
                n == name and not iss for n, iss in calls[i + 1:]
            ):
                self.stat_update("call_corrected_later")
                continue
            issues += call_issues

        if self.annotate_key:
            doc.metadata[self.annotate_key] = [list(x) for x in issues]
        kinds = list(dict.fromkeys(kind for kind, _, _ in issues))
        for kind in kinds:
            self.stat_update(kind)
        dropping = [k for k in kinds if k in self.drop_on]
        if dropping:
            return False, dropping[0]
        return True
