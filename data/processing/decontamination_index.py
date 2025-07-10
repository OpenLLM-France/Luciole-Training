from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontIndexer
import os

"""
Decontamination Indexing Script
List of tasks available on lighteval:
- https://huggingface.co/docs/lighteval/available-tasks
- https://github.com/huggingface/lighteval/blob/main/community_tasks/french_evals.py
- https://github.com/huggingface/lighteval/blob/main/community_tasks/arabic_evals.py
- https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/multilingual/tasks.py

Languages we are interested n:
- English (en)
- French (fr)
- Arabic (ar)
- Spanish (es)
- Portuguese (pt)
- Italian (it)
- German (de)
- Dutch (nl)
- Basque (eu)
- Catalan (ca)
- French regional languages 
"""

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")
if not MAIN_PATH:
    raise RuntimeError("Environment variable 'OpenLLM_OUTPUT' is not set or is empty.")
DATA_PATH = os.path.join(MAIN_PATH, "data/raw_data/full_datasets")

EN_TASKS = [
    "lighteval|openbookqa",
    "lighteval|piqa",
    "leaderboard|arc:challenge",
    "lighteval|arc:easy",
    "lighteval|triviaqa",
    "helm|boolq",
    "leaderboard|hellaswag",
    "leaderboard|winogrande",
]

FR_TASKS = [
    "lighteval|mlmm_arc_fra_cf:challenge",
    "lighteval|mintaka_fra",
    "lighteval|belebele_fra_Latn_cf",
    "lighteval|fquadv2_fra",
    "lighteval|xcodah_fra_cf",
    "lighteval|xcsqa_fra_cf",
    "lighteval|mlmm_hellaswag_fra_cf",
    "lighteval|xnli2.0_fra_cf",
]

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    config = NGramsDecontConfig()

    # ENGLISH
    pipeline = [
        NGramsDecontIndexer(
            output_folder=f"{DATA_PATH}/decontamination_index/data/en",
            config=config,
            lighteval_tasks=EN_TASKS,
        ),
        NGramsDecontIndexer(
            output_folder=f"{DATA_PATH}/decontamination_index/data/fr",
            config=config,
            lighteval_tasks=FR_TASKS,
            language="fr",
            custom_lighteval_tasks="lighteval.tasks.multilingual.tasks",
        ),
    ]

    executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/decontamination_index/logs",
        job_name="decontamination_index",
        tasks=1,
        cpus_per_task=2,
        partition="prepost",
        time="02:00:00",
    )

    executor.run()
