import os

import lighteval.tasks.default_prompts as prompt
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)

from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm

from lighteval.tasks.default_prompts import LETTER_INDICES

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = []
TASKS_TABLE.extend(ML_TASKS_TABLE)

all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

SUBSETS = ["different", "word by word", "similar"]

# Idiomatic Expressions tasks
idiomatic_expressions_tasks = [
    LightevalTaskConfig(
        name=f"idiomatic_expressions_{formulation.name.lower()}:{subset.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": "Quelle est l'expression idiomatique parmi les 4 propositions suivantes?",
                "choices": [
                    line["masked sentences"].replace("< ...>", line["answer A"]),
                    line["masked sentences"].replace("< ...>", line["answer B"]),
                    line["masked sentences"].replace("< ...>", line["answer C"]),
                    line["masked sentences"].replace("< ...>", line["answer D"]),
                    ],
                "gold_idx": int(line["answer"]) - 1
                if line["answer"].isdigit()
                else LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="charlotte9901/idiomatic_expressions",
        hf_subset=subset,
        evaluation_splits=("train",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],)
    )
    for subset in SUBSETS
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(idiomatic_expressions_tasks)