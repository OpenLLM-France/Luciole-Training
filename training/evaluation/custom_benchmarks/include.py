from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from functools import partial
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation

TASKS_TABLE = []

LANGUAGES = [
    "Basque",
    "Dutch",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
]  # Other languages are available, see https://huggingface.co/datasets/CohereLabs/include-base-44

SUBSETS = [
    "Applied Science",
    "Arts & Humanities",
    "Business & Commerce",
    "Driving License",
    "General knowledge",
    "Health oriented education",
    "Marine License",
    "Medical License",
    "Professional certification",
    "STEM",
    "Social Science",
]

# INCLUDE original


def prompt(line, task_name: str = None) -> Doc:
    question = line["question"]
    option_a = line["option_a"]
    option_b = line["option_b"]
    option_c = line["option_c"]
    option_d = line["option_d"]

    query = f"{question.strip()}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:"
    choices = ["A", "B", "C", "D"]
    gold_idx = line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_idx,
    )


TASKS_TABLE.extend(
    [
        LightevalTaskConfig(
            name=f"include_{language.lower()}:{subset.lower().replace(' & ', '_').replace(' ', '_')}",
            prompt_function=prompt,
            suite=["custom"],
            hf_repo="CohereLabs/include-base-44",
            hf_subset=language,
            hf_filter=partial(lambda x, subset: x["domain"] == subset, subset=subset),
            evaluation_splits=("test",),
            few_shots_split="validation",
            metrics=[Metrics.loglikelihood_acc],
        )
        for language in LANGUAGES
        for subset in SUBSETS
    ]
)

# INCLUDE MCQ tasks

all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]


def process_line(line):
    question = line["question"].strip()
    choices = [
        line["option_a"],
        line["option_b"],
        line["option_c"],
        line["option_d"],
    ]
    gold_idx = line["answer"]
    mcq_input = {
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
    }
    return mcq_input


def custom_filter(line, subset, regional_feature):
    if regional_feature == "all":
        return line["domain"] == subset
    elif regional_feature == "agnostic":
        return (line["domain"] == subset) & (line["regional_feature"] == "agnostic")
    else:  # non-agnostic
        return (line["domain"] == subset) & (line["regional_feature"] != "agnostic")


TASKS_TABLE.extend(
    [
        LightevalTaskConfig(
            name=f"include_{language.lower()}_mcq_{formulation.name.lower()}:{subset.lower().replace(' & ', '_').replace(' ', '_')}_{regional_feature.replace(' ', '_')}",
            prompt_function=get_mcq_prompt_function(
                Language.FRENCH,
                process_line,
                formulation=formulation,
            ),
            suite=["custom"],
            hf_repo="CohereLabs/include-base-44",
            hf_subset=language,
            hf_filter=partial(
                custom_filter, subset=subset, regional_feature=regional_feature
            ),
            evaluation_splits=("test",),
            few_shots_split="validation",
            metrics=get_metrics_for_formulation(
                formulation,
                [
                    LogLikelihoodAccMetric(),
                    LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                ],
            ),
        )
        for language in LANGUAGES
        for subset in SUBSETS
        for formulation in all_qa_formulations
        for regional_feature in ["all", "agnostic", "non-agnostic"]
    ]
)

print(f"Total tasks registered: {len(TASKS_TABLE)}")
print("Tasks:")
for task in TASKS_TABLE:
    print(f"  {task.name}")
