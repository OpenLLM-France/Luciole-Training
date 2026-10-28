from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the local filesystem. The content is returned with line numbers, one line per row.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read.",
                }
            },
            "required": ["path"],
        },
    },
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace an exact snippet of a file with new text. `old_string` must "
            "match the file exactly, including indentation, and appear only once "
            "in it. Read the file before editing it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file, overwriting it if it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content of the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

TOOLS = [READ_FILE_TOOL, EDIT_FILE_TOOL, WRITE_FILE_TOOL]

USER_TEMPLATES = [
    "{instruction}\n\nThe file is `{path}`.",
    "In `{path}`: {instruction}",
    "{instruction} (file: `{path}`)",
    "Edit `{path}`. {instruction}",
]

# Commit subjects that say nothing about the change.
BAD_INSTRUCTIONS = (
    "update",
    "fix",
    "wip",
    "merge",
    "cleanup",
    "clean up",
    "refactor",
    "changes",
    "minor changes",
    "small fix",
    "typo",
)


def number_lines(text: str) -> str:
    # Same layout as `cat -n`: right-aligned line number, a tab, the line.
    return "\n".join(
        f"{i:6}\t{line}" for i, line in enumerate(text.split("\n"), start=1)
    )


def build_edits(old: str, new: str, merge_gap: int = 3):
    """Turn a before/after pair into a list of (old_string, new_string) edits.

    The edits are applied one after another, top to bottom, with an exact
    single-occurrence replacement. Each hunk is widened with surrounding lines
    until its old_string is unique in the current file (and never empty, as
    pure insertions need an anchor). The context never reaches into the
    previous hunk, whose old text is already gone. Returns None when an edit
    cannot be made unique or the edits do not rebuild ``new``.
    """
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    opcodes = difflib.SequenceMatcher(
        None, old_lines, new_lines, autojunk=False
    ).get_opcodes()

    # Group the changed opcodes, merging those separated by <= merge_gap lines.
    hunks = []  # [i1, i2, j1, j2] spans in old / new lines
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if hunks and i1 - hunks[-1][1] <= merge_gap:
            hunks[-1][1], hunks[-1][3] = i2, j2
        else:
            hunks.append([i1, i2, j1, j2])

    current = old
    edits = []
    prev_end = 0  # end of the previous hunk, in old lines
    for k, (i1, i2, j1, j2) in enumerate(hunks):
        next_start = hunks[k + 1][0] if k + 1 < len(hunks) else len(old_lines)
        before, after = 0, 0
        while True:
            old_string = "".join(old_lines[i1 - before : i2 + after])
            new_string = "".join(
                old_lines[i1 - before : i1] + new_lines[j1:j2] + old_lines[i2 : i2 + after]
            )
            if old_string.strip() and current.count(old_string) == 1:
                break
            can_before = i1 - before > prev_end
            can_after = i2 + after < next_start
            if not (can_before or can_after):
                return None
            if can_before:
                before += 1
            if can_after:
                after += 1
        current = current.replace(old_string, new_string, 1)
        edits.append((old_string, new_string))
        prev_end = i2

    if current != new:
        return None
    return edits


def filter_and_format(
    data,
    rank: int = 0,
    world_size: int = 1,
    max_chars: int = 32_000,
    max_hunks: int = 5,
    write_ratio: float = 0.6,
    allow_new_files: bool = False,
):
    import difflib
    import random

    for doc in data:
        meta = doc.metadata
        path = meta["old_file"]
        old, new = meta["old_contents"], meta["new_contents"]
        instruction = meta["instruction"].strip()

        if path != meta["new_file"]:  # renames
            continue
        new_file = not old.strip()
        if (new_file and not allow_new_files) or not new.strip():
            continue
        if len(old) > max_chars or len(new) > max_chars:
            continue
        if "".join(old.split()) == "".join(new.split()):  # whitespace-only
            continue
        if len(instruction) < 10 or instruction.lower().rstrip(".") in BAD_INSTRUCTIONS:
            continue

        rng = random.Random(str(doc.id))
        user = rng.choice(USER_TEMPLATES).format(instruction=instruction, path=path)
        messages = [{"role": "user", "content": user}]

        if new_file:
            # Nothing to read: the file is created in one call.
            calls = [("write_file", {"path": path, "content": new})]
        else:
            messages += [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": path}},
                        }
                    ],
                },
                {"role": "tool", "content": number_lines(old)},
            ]
            # Mostly rewritten files are written back whole, the rest is edited.
            changed = 1 - difflib.SequenceMatcher(
                None, old.splitlines(), new.splitlines(), autojunk=False
            ).ratio()
            if changed >= write_ratio:
                calls = [("write_file", {"path": path, "content": new})]
            else:
                edits = build_edits(old, new)
                if not edits or len(edits) > max_hunks:
                    continue
                calls = [
                    ("edit_file", {"path": path, "old_string": o, "new_string": n})
                    for o, n in edits
                ]

        for name, arguments in calls:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": f"The file {path} has been {'created' if new_file else 'updated'}.",
                }
            )

        tools = list(TOOLS)
        rng.shuffle(tools)
        doc.metadata = {
            "dataset": meta.get("dataset"),
            "tools": tools,
            "messages": messages,
        }
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "nuprl/EditPackFT",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        filter_and_format,
        NemoRLFormat(),
        # Drops the trailing tool result: the sample ends on the last edit.
        check_last_message,
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": False, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/editpackft_oaiformat/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/editpackft_oaiformat",
            output_filename="data/editpackft_oaiformat/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/editpackft_oaiformat/logs",
        job_name="editpackft_oaiformat",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
