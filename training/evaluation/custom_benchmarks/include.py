from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from functools import partial

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


### Grammar Task
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

print(f"Total tasks registered: {len(TASKS_TABLE)}")
print("Tasks:")
for task in TASKS_TABLE:
    print(f"  {task.name}")
