from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    hub_adapter,
    FilterChinese,
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
    NemoRLFormat,
)
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

# The dataset ships no tool schema -- the traces only wrap raw Python in
# <tool_call> tags -- and neither the card nor the NeMo-Skills docs publish the
# one the generation harness used. This rebuilds it from what the traces show:
# self-contained snippets whose printed output (or last expression, notebook
# style) comes back in an ```output``` block. Key order follows the other
# datasets, see convert_to_openai_format in jupyter_agent.py.
PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run a self-contained snippet of Python code and return "
        "what it prints, plus the value of its last expression. Use it for "
        "exact arithmetic, symbolic manipulation and numerical checks instead "
        "of computing by hand.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
            },
            "required": ["code"],
        },
    },
}


class TIRFormat(BaseFilter):
    """Turn a flat TIR solution into a multi-turn tool-calling conversation.

    The whole tool-integrated reasoning loop happens *inside* one <think>
    block, with the final answer after it:

        <think> reasoning
          <tool_call> code </tool_call>
          ```output ... ```
          ```system Remaining code executions: N. ... ```
          reasoning ... (repeats)
        </think> final answer

    Kept as one turn the model would learn to write its own execution outputs,
    so each step is split into an assistant turn (reasoning + tool call) and a
    tool turn (the real output), which is also the shape the model needs at
    inference time.
    """

    name = "🐍 TIR Format"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool:
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "nvidia/OpenMathReasoning")
        import re

        # One TIR step. The ```system``` block only advertises the harness'
        # execution budget, which does not exist at inference time: it is
        # matched so it gets consumed, but never kept.
        step_pattern = re.compile(
            r"<tool_call>(?P<code>.*?)</tool_call>\s*"
            r"```output(?P<output>.*?)```"
            r"(?:\s*```system.*?```)?",
            re.DOTALL,
        )

        solution = doc.metadata.pop("generated_solution", "") or ""
        # Truncated generations leave a dangling <tool_call> that would parse as
        # reasoning text.
        if solution.count("<tool_call>") != solution.count("</tool_call>"):
            return False, "unclosed_tool_call"

        split = re.match(r"\s*<think>(?P<reasoning>.*)</think>(?P<answer>.*)", solution, re.DOTALL)
        if not split:
            return False, "no_think_block"
        reasoning, answer = split.group("reasoning"), split.group("answer").strip()
        if not answer:
            return False, "empty_answer"

        steps = list(step_pattern.finditer(reasoning))
        if not steps:
            return False, "no_tool_call"

        messages = [{"role": "user", "content": doc.metadata["problem"]}]
        cursor = 0
        for step in steps:
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": reasoning[cursor : step.start()].strip(),
                    "content": "",
                    "tool_calls": [
                        {
                            "name": PYTHON_TOOL["function"]["name"],
                            "arguments": {"code": step.group("code").strip()},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "content": step.group("output").strip()})
            cursor = step.end()
        # Everything after the last execution: the tail of the reasoning, then
        # the answer written outside the <think> block.
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": reasoning[cursor:].strip(),
                "content": answer,
            }
        )

        doc.metadata["messages"] = messages
        # Leave tools as a list of dicts; add_system_prompt bakes them into the
        # system message and json.dumps them for a load_dataset-friendly column.
        doc.metadata["tools"] = [PYTHON_TOOL]
        return True


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        ParquetReader(
            "hf://datasets/nvidia/OpenMathReasoning/data/",
            glob_pattern="tir-*.parquet",
            adapter=instruct_adapter,
        ),
        TIRFormat(
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/openmathreasoning_tir/tir_parsing_error"
            ),
        ),
        partial(add_system_prompt, tokenizer=tokenizer),
        check_last_message,
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        # DeepSeek-R1 code-switches to Chinese in its reasoning often enough to
        # be worth filtering here.
        FilterChinese(
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/openmathreasoning_tir/chinese_heavy"
            ),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/openmathreasoning_tir",
            output_filename="data/openmathreasoning_tir/${rank}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{DATA_PATH}/openmathreasoning_tir/logs",
        job_name="openmathreasoning_tir",
        tasks=32,
        time="02:00:00",
        qos="qos_cpu-t3",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
