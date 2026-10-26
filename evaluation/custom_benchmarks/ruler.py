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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def ruler_prompt(line, task_name: str = None):
    """RULER prompt that keeps the answer prefix out of the user turn.

    Upstream ``default_prompts.ruler`` concatenates the answer prefix onto the user
    query; once a chat template is applied that leaves it dangling at the end of the
    user turn, and a chat model opens a fresh assistant turn and ignores it. Here the
    answer prefix is carried in ``specific["assistant_prefix"]`` so the prompt manager
    uses it as an assistant-turn prefill for chat models (matching NVIDIA's reference
    RULER setup). Completion (non-chat-template) models still receive it appended to
    the prompt exactly as before.
    """
    return Doc(
        task_name=task_name,
        query=line["input"],
        instruction=None,
        choices=line["outputs"],
        gold_index=list(range(len(line["outputs"]))),
        specific={"assistant_prefix": line.get("answer_prefix", "")},
    )


subsets = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

lengths = [131072, 65536, 32768, 16384, 8192, 4096]

task_configs = []

for subset in subsets:
    for length in lengths:
        if subset == "qa_2" and length == 4096:
            continue  # Dataset not available due to a bug during dataset generation
        task_configs.append(
            LightevalTaskConfig(
                name=f"ruler_{length}:{subset}",
                suite=["custom"],
                prompt_function=ruler_prompt,
                hf_repo="OpenLLM-France/RULER-luciole_tokenizer_128k-arab-regional_v2",
                hf_subset=f"{subset}-{length}",
                hf_avail_splits=["validation"],
                evaluation_splits=["validation"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=128
                if "niah" in subset
                else 30
                if subset == "vt"
                else 120
                if subset == "cwe"
                else 50,
                metrics=[Metrics.ruler_match_any]
                if subset in ["qa_1", "qa_2"]
                else [Metrics.ruler_match_all],
                stop_sequence=None,
            )
        )

TASKS_TABLE = task_configs
