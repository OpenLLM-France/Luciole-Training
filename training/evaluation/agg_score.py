import os
import warnings
from utils import read_experiment_results, task_group_mapping
import pandas as pd
import argparse


def read_info_baseline():
    df_info = pd.read_json("nb_answers_per_questions.jsonl", lines=True)
    df_info["random"] = 1.0 / df_info["num_classes"]
    task_info_mapping = df_info.set_index("task").to_dict(orient="index")
    return task_info_mapping


def get_info(task, task_info_mapping):
    if task == "all":
        return {}
    key_full = task.split("|")[1]
    key_base = key_full.split(":")[0]

    task_infos = task_info_mapping.get(key_full)
    if task_infos is None:
        task_infos = task_info_mapping.get(key_base)
    if task_infos is None:
        warnings.warn(f"No info found for task '{task.split('|')[1].split(':')[0]}'")
        return {}
    return task_infos


def normalize_within_range(value, lower_bound, higher_bound):
    if value < lower_bound:
        return 0
    else:
        return (value - lower_bound) / (higher_bound - lower_bound) * 100


def calculate_agg_score(df, task_metric_list, output_path=None, verbose=True):
    task_metric_set = set(task_metric_list)
    df = df[df[["task", "metric"]].apply(tuple, axis=1).isin(task_metric_set)]
    # Load task info mapping
    task_info_mapping = read_info_baseline()
    df_info = (
        df["task"].apply(lambda t: get_info(t, task_info_mapping)).apply(pd.Series)
    )
    df = df.join(df_info)
    df["norm_score"] = df.apply(
        lambda x: normalize_within_range(x["score"], x["random"], 1.0), axis=1
    )
    # Step 1: Group by ['task', 'metric', 'expe_name'], take row with max tokens
    group_df = df.loc[df.groupby(["task", "metric", "expe_name"])["tokens"].idxmax()]
    # Step 2: Group by ['expe_name', 'task_type', 'language'], mean of norm_score
    group_df = (
        group_df.groupby(["expe_name", "task_type", "language"])["norm_score"]
        .mean()
        .reset_index()
    )
    group_df = group_df.sort_values(
        ["language", "task_type", "norm_score"], ascending=False
    )
    if verbose:
        print("Task type agg:")
        print(group_df)
    if output_path is not None:
        group_df.to_csv(os.path.join(output_path, "task_type.csv"))
    # Step 3: Group by ['expe_name', 'language'], mean of previous results
    group_df = (
        group_df.groupby(["expe_name", "language"])["norm_score"].mean().reset_index()
    )
    group_df = group_df.sort_values(["language", "norm_score"], ascending=False)
    if verbose:
        print("Language agg:")
        print(group_df)
    if output_path is not None:
        group_df.to_csv(os.path.join(output_path, "language.csv"))
    return None


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
    args = parser.parse_args()

    df = pd.concat([read_experiment_results(path) for path in args.experiment_path])

    for g in args.group:
        os.makedirs(os.path.join(args.output_path, g), exist_ok=True)
        calculate_agg_score(
            df, task_group_mapping[g], output_path=os.path.join(args.output_path, g)
        )
