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

MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

AR_TASKS = [
    "lighteval|exams_ara_cf",
    "lighteval|alghafa_arc_ara_cf",
    "lighteval|alghafa_sciqa_ara_cf",
    "lighteval|belebele_arb_Arab_cf",
    "lighteval|soqal_ara_cf",
    "lighteval|mlqa_ara",
    "lighteval|tydiqa_ara",
    "lighteval|alghafa_race_ara_cf",
    "lighteval|arcd_ara",
    "lighteval|xcodah_ara_cf",
    "lighteval|alghafa_piqa_ara_cf",
    "lighteval|xcsqa_ara_cf",
    "lighteval|xnli2.0_ara_cf",
    "lighteval|mlmm_hellaswag_ara_cf",
    "lighteval|xstory_cloze_ara_cf",
] + ["lighteval|mmlu_ara_cf:" + subset for subset in MMLU_SUBSETS]

CA_TASKS = [
    "lighteval|mlmm_hellaswag_cat_cf",
    "lighteval|belebele_cat_Latn_cf",
    "lighteval|mlmm_arc_cat_cf:challenge",
    "lighteval|mlmm_truthfulqa_cat_cf",
] + ["lighteval|mlmm_mmlu_cat_cf:" + subset for subset in MMLU_SUBSETS]

DE_TASKS = [
    "lighteval|xnli2.0_deu_cf",
    "lighteval|pawsx_deu_cf",
    "lighteval|mlmm_hellaswag_deu_cf",
    "lighteval|germanquad_deu",
    "lighteval|mlqa_deu",
    "lighteval|belebele_deu_Latn_cf",
    "lighteval|mlmm_arc_deu_cf:challenge",
    "lighteval|lumi_arc_deu_cf:challenge",
    "lighteval|mlmm_truthfulqa_deu_cf",
    "lighteval|xcsqa_deu_cf",
    "lighteval|xcodah_deu_cf",
    "lighteval|mkqa_deu",
    "lighteval|mintaka_deu",
] + ["lighteval|meta_mmlu_deu_cf:" + subset for subset in MMLU_SUBSETS]

EU_TASKS = [
    "lighteval|mlmm_hellaswag_eus_cf",
    "lighteval|xstory_cloze_eus_cf",
    "lighteval|mlmm_truthfulqa_eus_cf",
    "lighteval|belebele_eus_Latn_cf",
] 

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

ES_TASKS = [
    "lighteval|xnli2.0_spa_cf",
    "lighteval|pawsx_spa_cf",
    "lighteval|mlmm_hellaswag_spa_cf",
    "lighteval|xquad_spa",
    "lighteval|squad_spa",
    "lighteval|mlqa_spa",
    "lighteval|belebele_spa_Latn_cf",
    "lighteval|mlmm_arc_spa_cf:challenge",
    "lighteval|lumi_arc_spa_cf:challenge",
    "lighteval|xcsqa_spa_cf",
    "lighteval|openbookqa_spa",
    "lighteval|xcodah_spa_cf",
    "lighteval|xstory_cloze_spa_cf",
    "lighteval|mkqa_spa",
    "lighteval|mintaka_spa",
] + ["lighteval|meta_mmlu_spa_cf:" + subset for subset in MMLU_SUBSETS]

FR_TASKS = [
    "lighteval|mlmm_arc_fra_cf:challenge",
    "lighteval|mintaka_fra",
    "lighteval|belebele_fra_Latn_cf",
    "lighteval|fquadv2_fra",
    "lighteval|xcodah_fra_cf",
    "lighteval|xcsqa_fra_cf",
    "lighteval|mlmm_hellaswag_fra_cf",
    "lighteval|xnli2.0_fra_cf",
    "lighteval|xwinograd_fra_cf",
    "lighteval|mkqa_fra",
    "lighteval|mlmm_truthfulqa_fra_cf",
] + ["lighteval|meta_mmlu_fra_cf:" + subset for subset in MMLU_SUBSETS]

IT_TASKS = [
    "lighteval|xcopa_ita_cf",
    "lighteval|mlmm_hellaswag_ita_cf",
    "lighteval|squad_ita",
    "lighteval|belebele_ita_Latn_cf",
    "lighteval|mlmm_arc_ita_cf:challenge",   
    "lighteval|lumi_arc_ita_cf:challenge", 
    "lighteval|mlmm_truthfulqa_ita_cf",
    "lighteval|m3exams_ita_cf",
    "lighteval|xcsqa_ita_cf",
    "lighteval|xcodah_ita_cf",
    "lighteval|mkqa_ita",
    "lighteval|mintaka_ita",
] + ["lighteval|meta_mmlu_ita_cf:" + subset for subset in MMLU_SUBSETS]

MATHS_TASKS = [
    "lighteval|math:algebra",
    "lighteval|math:counting_and_probability",
    "lighteval|math:geometry",
    "lighteval|math:intermediate_algebra",
    "lighteval|math:number_theory",
    "lighteval|math:prealgebra",
    "lighteval|math:precalculus",
    "lighteval|math_cot:algebra",
    'lighteval|math_cot:counting_and_probability',
    "lighteval|math_cot:geometry",
    "lighteval|math_cot:intermediate_algebra",
    "lighteval|math_cot:number_theory",
    "lighteval|math_cot:prealgebra",
    "lighteval|math_cot:precalculus",
    "lighteval|gsm8K",
    "lighteval|lambada:openai:fr",
]

MGSM_TASKS = [
    "lighteval|mgsm_fra", 
    "lighteval|mgsm_spa",
    "lighteval|mgsm_eng",
    "lighteval|mgsm_deu",   
]

NL_TASKS = [
    "lighteval|mlmm_hellaswag_nld_cf",
    "lighteval|mlmm_arc_nld_cf:challenge",
    "lighteval|mlmm_truthfulqa_nld_cf",
    "lighteval|xcsqa_nld_cf",
    "lighteval|xcodah_nld_cf",
    "lighteval|belebele_nld_Latn_cf",
] + ["lighteval|mlmm_mmlu_nld_cf:" + subset for subset in MMLU_SUBSETS]

PT_TASKS = [
    "lighteval|mlmm_hellaswag_por_cf",
    "lighteval|faquad_por",
    "lighteval|belebele_por_Latn_cf",
    "lighteval|lumi_arc_por_lumi_cd:challenge",
    "lighteval|mlmm_truthfulqa_por_cf",
    "lighteval|m3exams_por_cf",
    "lighteval|xcsqa_por_cf",
    "lighteval|oab_exams_por_cf",
    "lighteval|enem_por_cf:2024",
    "lighteval|xcodah_por_cf",
    "lighteval|xwinograd_por_cf",
    "lighteval|mkqa_por",
    "lighteval|mintaka_por",    
] + ["lighteval|meta_mmlu_por_cf:" + subset for subset in MMLU_SUBSETS]

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
