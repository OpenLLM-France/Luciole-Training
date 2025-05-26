from pathlib import Path
import json
import pandas as pd
import re
import json
import warnings

task_group_mapping = {
    "mmlu": [
        ("lighteval|meta_mmlu_fra_cf:abstract_algebra|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:anatomy|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:astronomy|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:business_ethics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:clinical_knowledge|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_biology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_chemistry|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_computer_science|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_mathematics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_medicine|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:college_physics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:computer_security|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:conceptual_physics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:econometrics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:electrical_engineering|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:elementary_mathematics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:formal_logic|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:global_facts|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_biology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_chemistry|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_computer_science|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_european_history|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_geography|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_government_and_politics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_macroeconomics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_mathematics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_microeconomics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_physics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_psychology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_statistics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_us_history|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:high_school_world_history|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:human_aging|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:human_sexuality|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:international_law|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:jurisprudence|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:logical_fallacies|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:machine_learning|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:management|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:marketing|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:medical_genetics|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:miscellaneous|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:moral_disputes|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:moral_scenarios|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:nutrition|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:philosophy|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:prehistory|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:professional_accounting|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:professional_law|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:professional_medicine|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:professional_psychology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:public_relations|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:security_studies|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:sociology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:us_foreign_policy|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:virology|0", "acc_norm"),
        ("lighteval|meta_mmlu_fra_cf:world_religions|0", "acc_norm"),
    ],
    "en": [
        ("helm|boolq|0", "pem"),
        ("lighteval|triviaqa|0", "qem"),
        ("lighteval|arc:easy|0", "acc"),
        ("lighteval|arc:easy|0", "acc_norm"),
        ("leaderboard|arc:challenge|0", "acc"),
        ("leaderboard|arc:challenge|0", "acc_norm"),
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|openbookqa|0", "acc_norm"),
        ("lighteval|piqa|0", "acc_norm"),
    ],
    "fr": [
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm"),
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm"),
        ("lighteval|xcodah_fra_cf|0", "acc_norm"),
        ("lighteval|xcsqa_fra_cf|0", "acc_norm"),
        ("lighteval|xnli2.0_fra_cf|0", "acc_norm"),
        ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"),
        ("lighteval|fquadv2_fra|0", "f1_fra"),
        ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"),
        ("lighteval|mintaka_fra|0", "f1_fra"),
    ],
}

df_info = pd.read_json("nb_answers_per_questions.jsonl", lines=True)
df_info['random'] = 1./df_info['num_classes']
task_info_mapping = df_info.fillna(0.).set_index("task").to_dict(orient="index")

def get_task_info(task):
    key_full = task.split('|')[1]
    key_base = key_full.split(':')[0]

    task_infos = task_info_mapping.get(key_full)
    if task_infos is None:
        task_infos = task_info_mapping.get(key_base)
    if task_infos is None:
        warnings.warn(f"No info found for task '{task.split('|')[1].split(':')[0]}'")
    return task_infos

def read_json_results(file):
    with open(file, "r") as file:
        data = json.load(file)
    model_name = data["config_general"]["model_name"]
    results = data["results"]
    df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="task")
    df["model_name"] = model_name
    match = re.match(r"results_(.*)\.json", file.name)
    df["timestamp"] = match.group(1) if match else None
    return df


def read_experiment_results(main_dir):
    main_dir = Path(main_dir)
    experiment_name = main_dir.name

    json_files = main_dir.rglob("results_*.json")  # recursively finds all .json files
    if json_files:
        df = pd.concat([read_json_results(file) for file in json_files], ignore_index=True)
    else:
        raise FileNotFoundError(f"No JSON result files found in {main_dir}")

    # Fix date and deduplicate
    df["timestamp"] = (
        df["timestamp"]
        .str.replace("T", " ")
        .str.replace(r"(\d{2})-(\d{2})-(\d{2}\.\d+)", r"\1:\2:\3", regex=True)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Post process some columns
    df["experiment_name"] = experiment_name
    df["step"] = df["model_name"].str.extract(r"--step_([0-9.]+)-")[0].astype(float)
    df["samples"] = (
        df["model_name"].str.extract(r"-consumed_samples_([0-9.]+)")[0].astype(float)
    )
    df["tokens"] = df["samples"] * 2048 / 10**9

    # Remove duplicates
    columns = ["task", "experiment_name", "step", "samples", "tokens"]
    df_sorted = df.sort_values(["timestamp", "tokens"])
    df = df_sorted.drop_duplicates(subset=columns, keep="last")
    return df