"""HotPotQA task on the original `hotpotqa/hotpot_qa` dataset.

Reuses the qa_f1 metric from sibling `longbench.py` (same normalize_answer +
token-F1 + max-over-golds machinery as lm-evaluation-harness `longbench_hotpotqa`).
"""

import os
import sys
import re
import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

sys.path.insert(0, os.path.dirname(__file__))

from longbench import qa_f1_metric, qa_f1_score  # noqa: E402


def hotpotqa_prompt_fn(line, task_name: str = None):
    context = line["context"]
    context_str = ""
    for i, (title, sentences) in enumerate(zip(context["title"], context["sentences"])):
        context_str += (
            f"Passage {i+1}:\n{title}\n"
            + "\n".join([sentence.strip() for sentence in sentences])
            + "\n\n"
        )

    query = (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n\n"
        f"{context_str}"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        f"Question: {line['question']}\n"
        "Answer:"
    )
    answers = [line["answer"]]
    return Doc(
        task_name=task_name,
        query=query,
        choices=answers,
        gold_index=list(range(len(answers))),
    )


hotpotqa = LightevalTaskConfig(
    name="hotpotqa_hotpotqa",
    suite=["custom"],
    prompt_function=hotpotqa_prompt_fn,
    hf_repo="OpenLLM-BPI/hotpotqa_subset",
    hf_subset="default",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32,
    stop_sequence=[],
    metrics=[qa_f1_metric],
    version=0,
)


# Pleais version


def hotpotqa_pleais_prompt_fn(line, task_name: str = None):
    question = line["question"]
    context = line["context"]
    context_str = ""
    for i, (title, sentences) in enumerate(zip(context["title"], context["sentences"])):
        context_str += (
            f"<|source_start|><source_id>{i+1} {title}: "
            + "".join(sentences)
            + "<|source_end|>\n"
        )

    query = f"<|query_start|>{question}<|query_end|>\n{context_str}<|language_start|>"
    answers = [line["answer"]]
    return Doc(
        task_name=task_name,
        query=query,
        choices=answers,
        gold_index=list(range(len(answers))),
    )


def pleais_postprocess_fn(generated_text):
    m = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", generated_text, re.DOTALL)
    if not m:
        return ""
    return re.sub(r"<ref name=.*?</ref>", "", m.group(1)).strip()


class _PleaisQAF1(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
        if not model_response.final_text:
            return 0.0
        cleaned = [pleais_postprocess_fn(t) for t in model_response.final_text]
        model_response.text_post_processed = cleaned
        prediction = cleaned[0]
        return max(
            (qa_f1_score(prediction, gt) for gt in doc.get_golds()),
            default=0.0,
        )


qa_f1_pleais_metric = SampleLevelMetric(
    metric_name="qa_f1_score",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=_PleaisQAF1(),
    corpus_level_fn=np.mean,
)

hotpotqa_pleais = LightevalTaskConfig(
    name="hotpotqa_hotpotqa_pleais",
    suite=["custom"],
    prompt_function=hotpotqa_pleais_prompt_fn,
    hf_repo="OpenLLM-BPI/hotpotqa_subset",
    hf_subset="default",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    stop_sequence=[],
    metrics=[qa_f1_pleais_metric],
    version=0,
)

TASKS_TABLE = [hotpotqa, hotpotqa_pleais]
