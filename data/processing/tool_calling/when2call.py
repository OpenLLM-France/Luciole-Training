from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from utils import (
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        tools = doc.metadata.get("tools", [])
        functions = [json.loads(tool) for tool in tools]
        for function in functions:
            # In When2Call, "required" is a sibling of "parameters"; the
            # OpenAI / BFCL schema expects it inside "parameters".
            if "required" in function:
                function.setdefault("parameters", {})["required"] = function.pop(
                    "required"
                )
        tools = [{"type": "function", "function": function} for function in functions]
        random.shuffle(tools)
        doc.metadata["tools"] = tools

        if any(
            "<tool_calls>" in message["content"] for message in doc.metadata["messages"]
        ):
            raise ValueError("Tool calls should not be in the messages")
        yield doc


def annotate_refusal(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        doc.metadata["refusal"] = "missing_argument"
        for word in ["sorry", "apologies", "apologize"]:
            if word in doc.metadata["messages"][-1]["content"].lower():
                doc.metadata["refusal"] = "apologies"
                break
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path


    pipeline = [
        HuggingFaceDatasetReader(
            "nvidia/When2Call",
            {"name": "train_sft", "split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        annotate_refusal,
        format_messages,
        # partial(replace_tool_name, rename_names=True, rename_params=False),
        NemoRLFormat(),
        check_last_message,
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/when2call_oaiformat/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/when2call_oaiformat",
            output_filename="data/when2call_oaiformat/${refusal}/${rank}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
            commit_message=args.commit,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/when2call_oaiformat/logs",
        job_name="when2call_oaiformat",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
