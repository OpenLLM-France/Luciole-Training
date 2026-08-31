from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language
from functools import partial


TASKS_TABLE = []

SUBSETS = [
    'mot réutilisé',
    'expression idiomatique',
    'mot emprunté',
    'verlan',
    'mot emprunté anglais',
    'mot modifié',
    'siglaison',
    'sens modifié'
]


# CORE FUNCTION (CONTEXT ONLY)

def get_choices_and_gold_from_line(line):
    sentence_context = line["French with context"]

    choices = [
        sentence_context.replace("< ...>", line["answer A"]),
        sentence_context.replace("< ...>", line["answer B"]),
        sentence_context.replace("< ...>", line["answer C"]),
        sentence_context.replace("< ...>", line["answer D"]),
    ]

    gold_idx = (
        int(line["answer"]) - 1
        if line["answer"].isdigit()
        else LETTER_INDICES.index(line["answer"])
    )

    return choices, gold_idx

# MCQ TASK

def process_line(line):
    question = "Quelle est la réponse correcte parmi les 4 propositions suivantes?"
    choices, gold_idx = get_choices_and_gold_from_line(line)

    return {
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
    }


all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

TASKS_TABLE.extend(
    [
        LightevalTaskConfig(
            name=f"french_youth_slang_mcq_context_{formulation.name.lower()}:{subset.lower().replace(' ', '_')}",
            prompt_function=get_mcq_prompt_function(
                Language.FRENCH,
                process_line,
                formulation=formulation,
            ),
            suite=["custom"],
            hf_repo="OpenLLM-BPI/french_youth_slang",
            hf_subset=subset,
            evaluation_splits=("train",),
            few_shots_split="train",
            metrics=get_metrics_for_formulation(
                formulation,
                [
                    LogLikelihoodAccMetric(),
                    LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                ],
            ),
        )
        for subset in SUBSETS
        for formulation in all_qa_formulations
    ]
)



# FIB TASK

def prompt_fn(line, task_name: str) -> Doc:
    choices, gold_idx = get_choices_and_gold_from_line(line)

    return Doc(
        task_name=task_name,
        query="", 
        choices=choices,
        gold_index=gold_idx,
    )


TASKS_TABLE.extend(
    [
        LightevalTaskConfig(
            name=f"french_youth_slang_fib_context:{subset.lower().replace(' ', '_')}",
            prompt_function=prompt_fn,
            suite=["custom"],
            hf_repo="OpenLLM-BPI/french_youth_slang",
            hf_subset=subset,
            evaluation_splits=("train",),
            few_shots_split="train",
            metrics=[Metrics.loglikelihood_acc],
        )
        for subset in SUBSETS
    ]
)


print(f"Total tasks registered: {len(TASKS_TABLE)}")
print("Tasks:")
for task in TASKS_TABLE:
    print(f"  {task.name}")