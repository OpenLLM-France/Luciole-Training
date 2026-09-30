#!/usr/bin/env python

from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from transformers import AutoTokenizer
from functools import partial
from utils import (
    hub_adapter,
    FilterChinese,
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
    NemoRLFormat,
)

def id_deduplication(data, rank: int = 0, world_size: int = 1):
    dedup_ids = set()
    for doc in data:
        id = doc.id
        if id in dedup_ids:
            continue
        dedup_ids.add(id)
        yield doc

def format_messages(data, rank: int = 0, world_size: int = 1):
    import json
    import re

    for doc in data:
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "OpenLLM-France/ReAct")
        messages = doc.metadata.pop("messages")
        tools = json.loads(doc.metadata.pop("tools"))
        tools = [tool for tool in tools if tool.get("function").get("name") != "submit_answer"]

        doc.metadata["thinking_dir"] = (
            "thinking" if doc.metadata.get("enable_thinking") else "non_thinking"
        )

        last_content = messages[-1]["content"]
        if "<tool_call>" not in last_content:
            # Full-text answer (e.g. PleAIs_RAG, which has no submit_answer
            # tool): leave the answer untouched.
            doc.metadata["messages"] = messages[1:]
            doc.metadata["tools"] = tools
            yield doc
            continue

        tool_calls = [
            json.loads(match.group(1).strip())
            for match in re.finditer(
                r"<tool_call>(.*?)</tool_call>", last_content, flags=re.DOTALL
            )
        ]
        if len(tool_calls) > 1 or tool_calls[0]["name"] != "submit_answer":
            continue

        answer = tool_calls[0]["arguments"]
        content = answer["detailed_answer"]
        supporting_facts = answer.get("supporting_facts") or []
        if supporting_facts:
            references = "\n".join(
                f"- [{fact['id']}] \"{fact['quote']}\"" if isinstance(fact, dict) else f'- "{fact}"'
                for fact in supporting_facts
            )
            content = f"{content}\n\nReferences:\n{references}"
        messages[-1]["content"] = content
        doc.metadata["messages"] = messages[1:]
        doc.metadata["tools"] = tools
        yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the per-episode draw."
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train")

    pipeline = [
        ParquetReader(
            "hf://datasets/OpenLLM-France/ReAct/data/",
            glob_pattern="**/*.parquet",
            adapter=instruct_adapter,
        ),
        id_deduplication,
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/react_postprocess",
            output_filename="data/react_postprocess/${dataset_name}/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/react_postprocess/logs",
        job_name="react_postprocess",
        tasks=14,
        time="01:00:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
