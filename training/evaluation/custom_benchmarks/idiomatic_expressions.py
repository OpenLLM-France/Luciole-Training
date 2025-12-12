from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
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

all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

SUBSETS = ["different", "word by word", "similar"]


def process_line(line, use_context, use_instruction):
    if use_instruction:
        question = (
            "Quelle est l'expression idiomatique parmi les 4 propositions suivantes?"
        )
    else:
        question = ""
    if use_context:
        sentence_context = line["French with context"]
    else:
        sentence_context = line["masked sentences"]
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

    mcq_input = {
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
    }
    return mcq_input


# Idiomatic Expressions tasks
idiomatic_expressions_tasks = [
    LightevalTaskConfig(
        name=f"idiomatic_expressions{'_instruct' if use_instruction else ''}{'_context' if use_context else ''}_{formulation.name.lower()}:{subset.lower().replace(' ', '_')}",
        prompt_function=get_mcq_prompt_function(
            Language.FRENCH,
            partial(
                process_line, use_context=use_context, use_instruction=use_instruction
            ),
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="OpenLLM-BPI/french_idiomatic_expressions",
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
    for use_context in [True, False]
    for use_instruction in [True, False]
]

TASKS_TABLE = idiomatic_expressions_tasks
print(f"Total tasks registered: {len(TASKS_TABLE)}")
print("Tasks:")
for task in TASKS_TABLE:
    print(f"  {task.name}")
