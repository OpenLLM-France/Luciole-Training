import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import argparse
import json
import warnings

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


def read_json_results(file):
    with open(file, "r") as file:
        data = json.load(file)
    model_name = data["config_general"]["model_name"]
    results = data["results"]
    df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="task")
    df["model_name"] = model_name
    return df


def read_experiment_results(main_dir):
    main_dir = Path(main_dir)
    experiment_name = main_dir.name

    json_files = main_dir.rglob("results_*.json")  # recursively finds all .json files

    dfs = []
    for file in json_files:
        df_part = read_json_results(file)
        match = re.match(r"results_(.*)\.json", file.name)
        df_part["timestamp"] = match.group(1) if match else None
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)

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
    return df


def plot_task(ax, df, task, metric, xlog=False, no_std=False):
    df = df[df["task"] == task]
    df = df.sort_values("tokens")

    pivot_df = df.pivot(index="tokens", columns="experiment_name", values=metric)
    stderr_df = df.pivot(
        index="tokens", columns="experiment_name", values=metric + "_stderr"
    )

    for col in pivot_df.columns:
        mean = pivot_df[col].dropna()
        stderr = stderr_df[col].dropna()

        # Align indices in case they differ slightly after dropna
        common_index = mean.index.intersection(stderr.index)
        mean = mean.loc[common_index]
        stderr = stderr.loc[common_index]

        ax.plot(mean.index, mean.values, marker="+", label=col, alpha=0.8)
        if not no_std:
            ax.fill_between(mean.index, mean - stderr, mean + stderr, alpha=0.1)

    task_infos = get_task_info(task)
    if task_infos is not None:
        xmin, xmax = ax.get_xlim()
        ax.hlines(task_infos['random'], xmin, xmax, colors='gray', linestyles='dashed', label="random")

    ax.set_xlabel("B tokens")
    ax.set_ylabel(metric)
    ax.set_title(task)
    if xlog:
        ax.set_xscale("log")

def plot_list_of_tasks(
    df, list_of_tasks_to_plot, output_file=None, title=None, xlog=False, no_std=False
):
    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    # Use a dictionary to avoid duplicate labels
    legend_dict = {}

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(axes[i], df, task, metric, xlog=xlog, no_std=no_std)

        handles, labels = axes[i].get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_dict[l] = h  # If duplicate, keeps last

    # Dedicated subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        legend_dict.values(),
        legend_dict.keys(),
        title="Experiment name",
        loc="center",
    )

    # Hide any other unused subplots if any
    for j in range(len(list_of_tasks_to_plot), len(axes) - 1):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        nargs="+",
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        choices=list(task_group_mapping.keys()),
        default=["en"],
        help="List of predefined groups of tasks you want to plot. You can add groups in the mapping if you want.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )
    parser.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--no_std", action="store_true", help="Remove std")
    args = parser.parse_args()
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    dfs = []
    for path in args.experiment_path:
        dfs.append(read_experiment_results(path))
    df = pd.concat(dfs)

    columns = ["task", "experiment_name", "step", "samples", "tokens"]
    df_sorted = df.sort_values("timestamp")
    df = df_sorted.drop_duplicates(subset=columns, keep="last")

    for g in args.group:
        output_file = (
            os.path.join(args.output_path, f"{g}.png") if args.output_path else None
        )
        plot_list_of_tasks(
            df,
            task_group_mapping[g],
            output_file=output_file,
            title=None,
            xlog=args.xlog,
            no_std=args.no_std
        )

    if not args.output_path:
        plt.show()
