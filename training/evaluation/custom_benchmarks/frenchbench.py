from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from functools import partial

TASKS_TABLE = []


def prompt(line, task_name: str = None, task_type: str = None) -> Doc:
    choices = [
        line["question"].replace("<...>", line["answerA"]),
        line["question"].replace("<...>", line["answerB"]),
        line["question"].replace("<...>", line["answerC"]),
        line["question"].replace("<...>", line["answerD"]),
    ]
    gold_idx = LETTER_INDICES.index(line["answer"])

    if task_type == "grammar":
        query = "La phrase suivante est correcte grammaticalement:\n"
    elif task_type == "vocab":
        query = "La phrase suivante est logique sémantiquement:\n"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
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
            name="french_bench_grammar",
            prompt_function=partial(prompt, task_type="grammar"),
            suite=["custom"],
            hf_repo="manu/french-bench-grammar-vocab-reading",
            hf_subset="default",
            evaluation_splits=("Grammar",),
            few_shots_split="Grammar",
            metrics=[Metrics.loglikelihood_acc],
        )
    ]
)

### Vocab Task
TASKS_TABLE.extend(
    [
        LightevalTaskConfig(
            name="frenchbench_vocab",
            prompt_function=partial(prompt, task_type="vocab"),
            suite=["custom"],
            hf_repo="manu/french-bench-grammar-vocab-reading",
            hf_subset="default",
            evaluation_splits=("Vocabulary",),
            few_shots_split="Vocabulary",
            metrics=[Metrics.loglikelihood_acc],
        )
    ]
)

print(f"Total tasks registered: {len(TASKS_TABLE)}")
print("Tasks:")
for task in TASKS_TABLE:
    print(f"  {task.name}")
