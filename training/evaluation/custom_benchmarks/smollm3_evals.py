# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval. We need to use custom tasks for these benchmarks because many of the pre-existing tasks in LightEval are designed for different configurations or base models and thus we must adapt the prompt and metrics to the zero-shot generative case.

Usage:

lighteval vllm "model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
    "custom|gsm_plus|0|0,custom|mixeval_hard|0|0" \
    --use-chat-template \
    --output-dir evals/ \
    --custom-tasks tasks.py \
    --save-details
"""
from functools import partial

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import winogrand_adapter
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language

TASKS_TABLE = []
TASKS_TABLE.extend(ML_TASKS_TABLE)

# ------------------
# BASE MODEL EVALS
# ------------------
qa_metrics = [
    loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
    loglikelihood_acc_metric(normalization=LogProbCharNorm()),
]
all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# ARC tasks
arc_tasks = [
    LightevalTaskConfig(
        name=f"arc_{formulation.name.lower()}:{subset.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="allenai/ai2_arc",
        hf_subset=f"ARC-{subset}",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in ["Easy", "Challenge"]
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(arc_tasks)

# BoolQ task
boolq_task = LightevalTaskConfig(
    name="boolq_cf",
    prompt_function=get_boolq_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "answer": line["answer"],
            "context": line["passage"],
        },
        formulation=CFFormulation(),
    ),
    suite=("custom",),
    hf_repo="google/boolq",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=5,
    stop_sequence=["\n"],
    metric=get_metrics_for_formulation(CFFormulation(), qa_metrics),
)
TASKS_TABLE.append(boolq_task)

# HellaSwag tasks
hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                "activity_label": line["activity_label"],
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["train", "validation"],
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(hellaswag_tasks)

# CommonsenseQA tasks
commonsense_qa_tasks = [
    LightevalTaskConfig(
        name=f"commonsenseqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": line["choices"]["label"].index(line["answerKey"].strip()),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(commonsense_qa_tasks)

# OpenBookQA tasks
openbook_qa_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(openbook_qa_tasks)

# Winogrande tasks
winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_{formulation.name.lower()}",
        suite=("custom",),
        prompt_function=get_continuation_prompt_function(
            Language.ENGLISH,
            partial(winogrand_adapter, Language.ENGLISH),
            formulation=formulation,
        ),
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        trust_dataset=True,
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=qa_metrics,
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(winogrande_tasks)

# PIQA tasks
piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["goal"],
                "choices": [line["sol1"], line["sol2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(piqa_tasks)

# SIQA tasks
siqa_tasks = [
    LightevalTaskConfig(
        name=f"siqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answerA"], line["answerB"], line["answerC"]],
                "gold_idx": int(line["label"]) - 1,
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/social_i_qa",
        hf_revision="53620e5841fb12b08e082485797e7021d3684ea2",
        hf_subset="default",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(siqa_tasks)

# MMLU tasks
# fmt: off
MMLU_SUBSETS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]
# fmt: on

mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in MMLU_SUBSETS
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_tasks)

# MMLU Pro tasks
mmlu_pro_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_pro_tasks)

# TriviaQA tasks
triviqa_tasks = [
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=("custom",),
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        generation_size=256,
        stop_sequence=("\n",),
        metric=[Metrics.quasi_exact_match_triviaqa],
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(triviqa_tasks)


# BBH tasks
def bbh_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="Question: " + line["input"] + "\nAnswer: ",
        choices=[line["target"]],
        gold_index=0,
    )


# fmt: off
BBH_SUBSETS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]
# fmt: on

bbh_tasks = [
    LightevalTaskConfig(
        name=f"bbh:{subset}",
        prompt_function=bbh_prompt,
        suite=["custom"],
        hf_repo="lighteval/big_bench_hard",
        hf_subset=subset,
        hf_revision="80610173426f05e6f1448f047e2db4840a7dd899",
        metric=[Metrics.exact_match],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        trust_dataset=True,
        stop_sequence=["Question:"],
    )
    for subset in BBH_SUBSETS
]
TASKS_TABLE.extend(bbh_tasks)

# GSM8K tasks
gsm8k_tasks = [
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=("custom",),
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.expr_gold_metric],
        generation_size=256,
        stop_sequence=["Question:"],
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(gsm8k_tasks)

# MATH tasks
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

math_tasks = [
    LightevalTaskConfig(
        name=f"math_cot:{config}",
        suite=("custom",),
        prompt_function=prompt.math_cot,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=config,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=4096,
        metric=[latex_gold_metric],
        stop_sequence=["\n"],
        trust_dataset=True,
        version=0,
    )
    for config in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]
TASKS_TABLE.extend(math_tasks)
