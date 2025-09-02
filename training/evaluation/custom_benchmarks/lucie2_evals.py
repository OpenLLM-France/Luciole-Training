# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author: Olivier Gouvert
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from datasets import load_dataset
import os
import yaml

SAMPLE_SUBSETS = [
    "gutenberg_fr",
    "wikimedia_fr",
    "opene-edition_fr",
    "gallica_press_fr",
]

# Build the path relative to the script
config_path = os.path.join(
    "~/OpenLLM-BPI-Training",
    "data",
    "tokenization",
    "run",
    "configs",
    "data_to_tokenize_v1.yaml",
)
config_path = os.path.expanduser(config_path)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# datasets = config["dataset_groups"][0]["datasets"]
# print(datasets)
datasets = {
    ds["name"]: ds["path"]
    for group in config["dataset_groups"]
    for ds in group["datasets"]
}

main_dir = os.path.join(os.getenv("OpenLLM_OUTPUT"), "benchmarks/lucie2")
os.makedirs(main_dir, exist_ok=True)

for subset in SAMPLE_SUBSETS:
    processed_path = os.path.join(main_dir, f"processed_data/{subset}")

    # Process dataset for each language if not already processed
    if not os.path.exists(processed_path):
        print(f"Loading {subset}")
        ds = load_dataset(
            "json",
            data_files=f"{datasets[subset]}/*00000.jsonl.gz",
            split="train[:1000]",
        )
        ds = ds.map(lambda x: {"text": x["text"][:2048]})
        os.makedirs(processed_path, exist_ok=True)
        ds.to_json(os.path.join(processed_path, "data.jsonl"))


def prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name, query=line["text"], gold_index=None, choices=None
    )  # see the_pile


class CustomSubsetTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_repo,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
            hf_repo=hf_repo,
            hf_subset="default",
            metric=[
                Metrics.word_perplexity,
                Metrics.byte_perplexity,
                Metrics.bits_per_byte,
            ],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split="train",
            few_shots_select="random_sampling",
            suite=["community"],
            generation_size=-1,
            stop_sequence=[],
        )


SUBSET_TASKS = [
    CustomSubsetTask(
        name=f"lucie2:{subset}",
        hf_repo=os.path.join(main_dir, f"processed_data/{subset}"),
    )
    for subset in SAMPLE_SUBSETS
]
TASKS_TABLE = SUBSET_TASKS
