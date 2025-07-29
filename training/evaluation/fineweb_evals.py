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
from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

SAMPLE_SUBSETS = [
    "arb_Arab",
    "deu_Latn",
    "fra_Latn",
    "ita_Latn",
    "nld_Latn",
    "por_Latn",
    "cat_Latn",
    "aai_Latn",
]

# LOAD FINEWEB 2
main_dir = os.path.join(os.getenv("OpenLLM_OUTPUT", "benchmarks/fineweb2"))
folder = snapshot_download(
    "HuggingFaceFW/fineweb-2",
    repo_type="dataset",
    local_dir=os.path.join(main_dir, "original_test_set"),
    allow_patterns=[f"data/{language}/test/*" for language in SAMPLE_SUBSETS],
)

# SPLIT
for language in SAMPLE_SUBSETS:
    output_path = os.path.join(main_dir, f"processed_data/{language}")
    if not os.path.exists(output_path):
        print(f"Loading {language}")
        ds = load_dataset(
            os.path.join(main_dir, f"original_test_set/data/{language}/test")
        )["train"]
        ds = ds.map(lambda x: {"text": x["text"][:1000]})
        ds.to_json(os.path.join(output_path, "data.jsonl"))


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
        name=f"fineweb2:{subset}",
        hf_repo=os.path.join(main_dir, f"processed_data/{subset}"),
    )
    for subset in SAMPLE_SUBSETS
]
TASKS_TABLE = SUBSET_TASKS
