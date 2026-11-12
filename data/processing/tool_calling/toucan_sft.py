from utils import create_parser, parse_args, create_executor
from datatrove.data import Document
from datatrove.pipeline.filters import LambdaFilter, LanguageFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.writers.disk_base import DiskWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    NemoRLFormat,
    ChatTokenCount,
    FT176_LANGUAGES,
)


class ToucanFilter(BaseFilter):
    """Convert Toucan (ms-swift agent format) messages to OpenAI-style messages.

    Toucan uses separate ``tool_call`` / ``tool_response`` turns, where the
    ``tool_call`` content is a Python repr ``{'name': ..., 'arguments': '<json>'}``.
    Consecutive ``tool_call`` turns are attached as ``tool_calls`` to the
    preceding assistant turn (an empty assistant turn is created if there is
    none), and ``tool_response`` turns become ``tool`` turns.

    Default mode: drop any conversation with a bad call (arguments that are not
    valid JSON, or a tool name that is not in ``tools``).

    ``self_correction=True``: keep ONLY the conversations that contain bad calls
    the model recovers from, i.e. every bad call
      - gets an error from the environment (``An error occurred when calling
        tool ...`` for invalid JSON, ``Tool X does not exists.`` for an unknown
        name), matched to its call by position within the call/response block;
      - is followed later in the conversation by a valid call.
    Bad calls are kept verbatim (invalid JSON arguments are inserted as the raw
    string). Conversations without any bad call are dropped (they are already in
    the default output).
    """

    name = "🦜 Toucan Format"

    JSON_ERROR = "An error occurred when calling tool"
    UNKNOWN_TOOL_ERROR = "does not exist"

    def __init__(self, self_correction: bool = False, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.self_correction = self_correction

    @staticmethod
    def parse_call(content, tool_names):
        """Return (status, name, arguments_str) with status in
        ok / unparsable / invalid_json / unknown_tool."""
        import ast
        import json

        try:
            call = ast.literal_eval(content)
            name, arguments = call["name"], call["arguments"]
        except Exception:
            return "unparsable", None, None
        try:
            arguments = json.dumps(json.loads(arguments), ensure_ascii=False)
        except Exception:
            return "invalid_json", name, arguments
        if name not in tool_names:
            return "unknown_tool", name, arguments
        return "ok", name, arguments

    def filter(self, doc: Document):
        import json
        import random

        doc.metadata.setdefault("dataset", "Agent-Ark/Toucan-1.5M")
        # Language annotation (from LanguageFilter) reported in the stats.
        if "language" in doc.metadata:
            self.stat_update(f"question_language_{doc.metadata['language']}")

        tools = json.loads(doc.metadata["tools"])
        tool_names = {t["function"]["name"] for t in tools}
        raw = json.loads(doc.metadata["messages"])
        calls = {
            i: self.parse_call(m.get("content") or "", tool_names)
            for i, m in enumerate(raw)
            if m["role"] == "tool_call"
        }

        if not self.self_correction:
            for status, _, _ in calls.values():
                if status in ("unparsable", "invalid_json"):
                    return False, "tool_call_parsing_error"
                if status == "unknown_tool":
                    return False, "tool_call_unknown_tool"
        else:
            if any(status == "unparsable" for status, _, _ in calls.values()):
                return False, "unparsable_call"
            bad = [i for i, (status, _, _) in calls.items() if status != "ok"]
            if not bad:
                return False, "no_bad_call"
            # Match each call to its response by position inside a block of
            # consecutive tool_call turns followed by consecutive tool_response turns.
            response_of = {}
            i = 0
            while i < len(raw):
                if raw[i]["role"] != "tool_call":
                    i += 1
                    continue
                block_calls = []
                while i < len(raw) and raw[i]["role"] == "tool_call":
                    block_calls.append(i)
                    i += 1
                block_responses = []
                while i < len(raw) and raw[i]["role"] == "tool_response":
                    block_responses.append(i)
                    i += 1
                if len(block_calls) != len(block_responses):
                    return False, "call_response_mismatch"
                response_of.update(zip(block_calls, block_responses))
            ok_positions = [i for i, (status, _, _) in calls.items() if status == "ok"]
            for i in bad:
                status = calls[i][0]
                response = raw[response_of[i]].get("content") or ""
                if status == "invalid_json" and not response.startswith(self.JSON_ERROR):
                    return False, "invalid_json_without_error"
                if status == "unknown_tool" and self.UNKNOWN_TOOL_ERROR not in response:
                    return False, "unknown_tool_without_error"
                if not any(j > i for j in ok_positions):
                    return False, "no_retry"
            for i in bad:
                self.stat_update(f"bad_call_{calls[i][0]}")

        random.shuffle(tools)
        doc.metadata["tools"] = tools

        messages = []
        for i, message in enumerate(raw):
            role, content = message["role"], message.get("content") or ""
            if role == "tool_call":
                _, name, arguments = calls[i]
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": ""})
                messages[-1].setdefault("tool_calls", []).append(
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            elif role == "tool_response":
                messages.append({"role": "tool", "content": content.strip()})
            else:
                messages.append({"role": role, "content": content})

        doc.metadata["messages"] = messages
        return True


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--self_correction",
        action="store_true",
        help="Keep only the conversations with bad tool calls the model recovers "
             "from (see ToucanFilter), written to toucan_sft_self_correction",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    NAME = "toucan_sft_self_correction" if args.self_correction else "toucan_sft"

    pipeline = [
        ParquetReader(
            "hf://datasets/Agent-Ark/Toucan-1.5M/SFT/",
            glob_pattern="*.parquet",
            # doc.text is the question so that LanguageFilter runs on it.
            text_key="question",
            id_key="uuid",
            adapter=instruct_adapter,
        ),
        # Adds metadata["language"] and metadata["language_score"] predicted on
        # the question, then keeps only the FT176_LANGUAGES (pretraining/utils.py).
        LanguageFilter(
            backend="ft176",
            label_only=True,
        ),
        LambdaFilter(
            lambda doc: doc.metadata["language"] in FT176_LANGUAGES,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/{NAME}/language_filter",
                output_filename="${rank}.jsonl",
            ),
        ),
        ToucanFilter(
            self_correction=args.self_correction,
            # In self_correction mode the excluded docs are the default output:
            # only their drop reasons are kept, in the stats.
            exclusion_writer=None if args.self_correction else JsonlWriter(
                f"{DATA_PATH}/{NAME}/parsing_error",
                output_filename="${filter_reason}/${rank}.jsonl",
            ),
        ),
        NemoRLFormat(),
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": False, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/{NAME}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/{NAME}",
            output_filename=f"data/sft/{NAME}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            commit_message=args.commit,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{NAME}/logs",
        job_name=NAME,
        tasks=3,
        time="01:00:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
        # Local multi-task runs fork workers that cannot re-import the
        # file-loaded utils modules (pretraining_utils): run tasks in-process.
        **({"workers": 1} if args.local else {}),
    )
    main_processing_executor.run()
