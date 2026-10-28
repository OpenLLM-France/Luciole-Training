"""Remove the spurious leading <think></think> of some DPO thinking subsets.

In these subsets of Luciole-PostTraining-Dataset-1.1, the system and user
messages start with "<think></think>", and so does the last assistant turn,
before its real "<think>...</think>". The chat template takes the reasoning
before the first "</think>", so the real reasoning of the last turn is dropped
(and chosen / rejected that only differ by it become identical). This script
removes the "<think></think>" at the start of every message.

The output mirrors the repo layout (``dpo_thinking/<subset>/00.jsonl``, with
only the ``chosen`` and ``rejected`` columns), so that the files can replace
the ones of the Hub with a single upload.
"""

from utils import create_parser, parse_args, create_executor, instruct_adapter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

SOURCE = "OpenLLM-France/Luciole-PostTraining-Dataset-1.1"
SUBSETS = [
    "nemotron-instruction-following-chat-v1",
    "nemotron-stem",
    "opencodereasoning",
]


def remove_leading_empty_think(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    """Strip "<think></think>" at the start of each chosen / rejected message."""
    EMPTY_THINK = "<think></think>"

    for doc in data:
        for key in ("chosen", "rejected"):
            for message in doc.metadata[key]:
                content = message.get("content")
                if isinstance(content, str) and content.startswith(EMPTY_THINK):
                    message["content"] = content[len(EMPTY_THINK) :]
        yield doc


def chosen_rejected_adapter(self, document):
    """Keep only the ``chosen`` and ``rejected`` columns, as in the Hub files."""
    return {
        "chosen": document.metadata["chosen"],
        "rejected": document.metadata["rejected"],
    }


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    for subset in SUBSETS:
        pipeline = [
            JsonlReader(
                f"hf://datasets/{SOURCE}/dpo_thinking/{subset}",
                glob_pattern="*.jsonl",
                adapter=instruct_adapter,
            ),
            remove_leading_empty_think,
            JsonlWriter(
                f"{DATA_PATH}/clean_double_reasoning/dpo_thinking/{subset}",
                output_filename="00.jsonl",
                compression=None,
                adapter=chosen_rejected_adapter,
            ),
        ]

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/clean_double_reasoning/logs/{subset}",
            job_name=f"clean_double_reasoning_{subset}",
            tasks=1,
            time="00:30:00",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
