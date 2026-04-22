from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from datasets import get_dataset_split_names
from transformers import AutoTokenizer
from utils import FilterChinese, apply_chat_template, instruct_adapter


def add_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import re

    _has_custom_instructions = False

    for document in data:
        system_prompt = document.metadata["chat_template_kwargs"]["custom_instructions"]
        if system_prompt != "" and not _has_custom_instructions:
            _has_custom_instructions = True
            print(f"\n>>> Found custom instructions in the dataset: {system_prompt}\n")

        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=False,
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)
        document.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + document.metadata["messages"]
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    splits = get_dataset_split_names("HuggingFaceTB/smoltalk2", "SFT")
    tool_splits = [
        "smolagents_toolcalling_traces_think",
        "hermes_function_calling_v1_no_think",
        "xlam_traces_no_think",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    # Add adapter that add empty text if the text field is None, to avoid skipping data
    for split in splits:
        if split in tool_splits:
            print(f"\n\n#### Ignoring tool split: {split}")
            continue
        print(f"\n\n#### Processing split: {split}")

        pipeline = [
            HuggingFaceDatasetReader(
                "HuggingFaceTB/smoltalk2",
                {"name": "SFT", "split": split},
                streaming=True,
                adapter=instruct_adapter,
            ),
            partial(add_system_prompt, tokenizer=tokenizer),
            partial(apply_chat_template, tokenizer=tokenizer),
            FilterChinese(
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/smoltalk2/{split}/chinese_heavy"
                ),
            ),
            JsonlWriter(
                f"{DATA_PATH}/smoltalk2/{split}/data",
                expand_metadata=True,
            ),
        ]
        if split == "smoltalk_multilingual8_Qwen3_32B_think":
            add_sampler_filter(pipeline, 0.3)
        elif split == "OpenThoughts3_1.2M_no_think_no_think":
            add_sampler_filter(pipeline, 0.4)
        elif split == "OpenHermes_2.5_no_think":
            add_sampler_filter(pipeline, 0.5)
        elif split == "smoltalk_smollm3_smol_magpie_ultra_no_think":
            add_sampler_filter(pipeline, 0.5)
        elif split == "OpenThoughts3_1.2M_think":
            add_sampler_filter(pipeline, 0.02)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/smoltalk2/{split}/logs",
            job_name="smoltalk2",
            tasks=1,
            time="00:30:00",
            partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
