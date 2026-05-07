# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
# Metric/prompt logic ported verbatim from
#   EleutherAI/lm-evaluation-harness :: lm_eval/tasks/longbench
#   (Copyright (c) 2023 THU-KEG & Zhipu AI)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""LongBench community tasks for lighteval.

Exact mirror of EleutherAI/lm-evaluation-harness's `longbench` task suite:
  - lm_eval/tasks/longbench/metrics.py        -> metric functions below
  - lm_eval/tasks/longbench/_generate_config.py (DATASETS, prompts, max_gen_toks)
  - lm_eval/tasks/longbench/<task>.yaml       -> LightevalTaskConfig entries

All 33 tasks (LongBench + LongBench-E) are generated. Tasks live in suite
"community" so call them as e.g. `community|longbench_hotpotqa|0|0`.

Optional dependencies (mirroring lm-eval-harness): jieba, fuzzywuzzy, rouge.
"""

import re
import string
from collections import Counter

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


try:
    import jieba
    from fuzzywuzzy import fuzz
    from rouge import Rouge
except ImportError as e:
    raise ImportError(
        "LongBench requires extra deps: pip install jieba fuzzywuzzy rouge"
    ) from e


# ======================================================================
# Metrics — verbatim port of lm_eval/tasks/longbench/metrics.py
# ======================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    global _rouge
    if "_rouge" not in globals():
        _rouge = Rouge()
    try:
        scores = _rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    return f1_score(pred_tokens, gold_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    pred_tokens = list(jieba.cut(prediction, cut_all=False))
    gold_tokens = list(jieba.cut(ground_truth, cut_all=False))
    pred_tokens = [normalize_zh_answer(t) for t in pred_tokens]
    gold_tokens = [normalize_zh_answer(t) for t in gold_tokens]
    pred_tokens = [t for t in pred_tokens if len(t) > 0]
    gold_tokens = [t for t in gold_tokens if len(t) > 0]
    return f1_score(pred_tokens, gold_tokens)


# ======================================================================
# lighteval SampleLevelMetric wrappers
# ======================================================================


class _MaxOverGolds(SampleLevelComputation):
    """Take the max of `metric_fn(pred, gt)` across gold answers.

    Mirrors lm-eval-harness `get_<metric>_score` helpers, where the prediction
    is `results[0].strip()` (or unstripped for code_sim) and the score is the
    max over `doc["answers"]`.
    """

    def __init__(self, metric_fn, strip_prediction=True, needs_all_classes=False):
        self.metric_fn = metric_fn
        self.strip_prediction = strip_prediction
        self.needs_all_classes = needs_all_classes

    def compute(self, doc, model_response, **kwargs):
        if not model_response.final_text:
            return 0.0
        prediction = model_response.final_text[0]
        if self.strip_prediction:
            prediction = prediction.strip()
        extra = {}
        if self.needs_all_classes:
            spec = doc.specific or {}
            extra["all_classes"] = list(spec.get("all_classes") or [])
        output = 0.0
        for ground_truth in doc.get_golds():
            output = max(output, self.metric_fn(prediction, ground_truth, **extra))
        return float(output)


def _make_metric(name, fn, strip=True, needs_all_classes=False):
    return SampleLevelMetric(
        metric_name=name,
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=_MaxOverGolds(
            fn, strip_prediction=strip, needs_all_classes=needs_all_classes
        ),
        corpus_level_fn=np.mean,
    )


qa_f1_metric = _make_metric("qa_f1_score", qa_f1_score)
qa_f1_zh_metric = _make_metric("qa_f1_zh_score", qa_f1_zh_score)
rouge_metric = _make_metric("rouge_score", rouge_score)
rouge_zh_metric = _make_metric("rouge_zh_score", rouge_zh_score)
classification_metric = _make_metric(
    "classification_score", classification_score, needs_all_classes=True
)
count_metric = _make_metric("count_score", count_score)
retrieval_metric = _make_metric("retrieval_score", retrieval_score)
retrieval_zh_metric = _make_metric("retrieval_zh_score", retrieval_zh_score)
# code_sim: do NOT strip prediction (lm-eval-harness comment).
code_sim_metric = _make_metric("code_sim_score", code_sim_score, strip=False)


# ======================================================================
# Prompts and per-task metadata — verbatim from _generate_config.py
# ======================================================================


DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{question}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{question}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {question}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{question}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n{question}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n{question}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{question}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is:",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{question}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is:',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{question}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:",
    "repobench-p": "Please complete the code given below. \n{context}{question}Next line of code:",
}


# (max_gen_toks, metric, stop_sequence) per base task — from _generate_config.py
# stop_sequence is ["\n"] for the few-shot tasks (trec/triviaqa/samsum/lsht), else [].
DATASET2META = {
    "narrativeqa": (128, qa_f1_metric, []),
    "qasper": (128, qa_f1_metric, []),
    "multifieldqa_en": (64, qa_f1_metric, []),
    "multifieldqa_zh": (64, qa_f1_zh_metric, []),
    "hotpotqa": (32, qa_f1_metric, []),
    "2wikimqa": (32, qa_f1_metric, []),
    "musique": (32, qa_f1_metric, []),
    "dureader": (128, rouge_zh_metric, []),
    "gov_report": (512, rouge_metric, []),
    "qmsum": (512, rouge_metric, []),
    "multi_news": (512, rouge_metric, []),
    "vcsum": (512, rouge_zh_metric, []),
    "trec": (64, classification_metric, ["\n"]),
    "triviaqa": (32, qa_f1_metric, ["\n"]),
    "samsum": (128, rouge_metric, ["\n"]),
    "lsht": (64, classification_metric, ["\n"]),
    "passage_count": (32, count_metric, []),
    "passage_retrieval_en": (32, retrieval_metric, []),
    "passage_retrieval_zh": (32, retrieval_zh_metric, []),
    "lcc": (64, code_sim_metric, []),
    "repobench-p": (64, code_sim_metric, []),
}


# Full list of HuggingFace dataset configs (base + LongBench-E variants),
# mirroring DATASETS in _generate_config.py.
DATASETS = [
    "2wikimqa",
    "2wikimqa_e",
    "dureader",
    "gov_report",
    "gov_report_e",
    "hotpotqa",
    "hotpotqa_e",
    "lcc",
    "lcc_e",
    "lsht",
    "multi_news",
    "multi_news_e",
    "multifieldqa_en",
    "multifieldqa_en_e",
    "multifieldqa_zh",
    "musique",
    "narrativeqa",
    "passage_count",
    "passage_count_e",
    "passage_retrieval_en",
    "passage_retrieval_en_e",
    "passage_retrieval_zh",
    "qasper",
    "qasper_e",
    "qmsum",
    "repobench-p",
    "repobench-p_e",
    "samsum",
    "samsum_e",
    "trec",
    "trec_e",
    "triviaqa",
    "triviaqa_e",
    "vcsum",
]


# ======================================================================
# Prompt function factory and task table generation
# ======================================================================


def _make_prompt_fn(prompt_template, base_name):
    needs_all_classes = base_name in ("trec", "lsht")

    def prompt_fn(line, task_name=None):
        query = prompt_template.format(
            context=line.get("context", ""),
            question=line.get("question", ""),
        )
        answers = list(line["answers"])
        specific = None
        if needs_all_classes:
            specific = {"all_classes": list(line.get("all_classes") or [])}
        return Doc(
            task_name=task_name,
            query=query,
            choices=answers,
            gold_index=list(range(len(answers))),
            specific=specific,
        )

    return prompt_fn


TASKS_TABLE = []
for _ds in DATASETS:
    _base = _ds[:-2] if _ds.endswith("_e") else _ds
    _max_toks, _metric, _stop = DATASET2META[_base]
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"longbench_{_ds}",
            suite=["community"],
            prompt_function=_make_prompt_fn(DATASET2PROMPT[_base], _base),
            hf_repo="Xnhyacinth/LongBench",
            hf_subset=_ds,
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=_max_toks,
            stop_sequence=list(_stop),
            metrics=[_metric],
            version=0,
        )
    )
