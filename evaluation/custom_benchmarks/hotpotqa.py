"""HotPotQA task on the original `hotpotqa/hotpot_qa` dataset.

Reuses the qa_f1 metric from sibling `longbench.py` (same normalize_answer +
token-F1 + max-over-golds machinery as lm-evaluation-harness `longbench_hotpotqa`).
Precision and recall of the F1-winning gold are reported alongside F1.
"""

import os
import sys
import re
import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

sys.path.insert(0, os.path.dirname(__file__))

from longbench import qa_f1_metric, qa_f1_prf_score  # noqa: E402


def hotpotqa_prompt_fn(line, task_name: str = None):
    context = line["context"]
    context_str = ""
    for i, (title, sentences) in enumerate(zip(context["title"], context["sentences"])):
        context_str += (
            f"Passage {i + 1}:\n{title}\n"
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
            f"<|source_start|><source_id>{i + 1} {title}: "
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


class _PleaisQAF1PRF(SampleLevelComputation):
    """Pleais postprocessing + {f1, precision, recall} of the best-F1 gold."""

    def compute(self, doc, model_response, **kwargs):
        zero = {"qa_f1_score": 0.0, "qa_precision_score": 0.0, "qa_recall_score": 0.0}
        if not model_response.final_text:
            return zero
        cleaned = [pleais_postprocess_fn(t) for t in model_response.final_text]
        model_response.text_post_processed = cleaned
        prediction = cleaned[0]
        best_p, best_r, best_f1 = 0.0, 0.0, 0.0
        for gt in doc.get_golds():
            p, r, f1 = qa_f1_prf_score(prediction, gt)
            if f1 > best_f1:
                best_p, best_r, best_f1 = p, r, f1
        return {
            "qa_f1_score": float(best_f1),
            "qa_precision_score": float(best_p),
            "qa_recall_score": float(best_r),
        }


_PLEAIS_KEYS = ["qa_f1_score", "qa_precision_score", "qa_recall_score"]

qa_f1_pleais_metric = SampleLevelMetricGrouping(
    metric_name=_PLEAIS_KEYS,
    higher_is_better=dict.fromkeys(_PLEAIS_KEYS, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=_PleaisQAF1PRF(),
    corpus_level_fn=dict.fromkeys(_PLEAIS_KEYS, np.mean),
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
