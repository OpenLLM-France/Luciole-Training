from utils import read_experiment_results
import io
import pandas as pd
import argparse


def read_info():
    with open("agg_tasks.jsonl") as f:
        content = "\n".join(line for line in f if not line.lstrip().startswith("//"))
    df_info = pd.read_json(io.StringIO(content), lines=True)
    df_info["random"] = (1.0 / df_info["num_classes"]).fillna(0.0)
    return df_info


_df_info = None


def get_info():
    global _df_info
    if _df_info is None:
        _df_info = read_info()
    return _df_info


def normalize_within_range(value, lower_bound, higher_bound):
    assert (
        lower_bound < higher_bound
        and value <= higher_bound
        and value >= min(0, lower_bound)
    ), f"Value {value} is out of bounds [{lower_bound}, {higher_bound}]"
    if value < lower_bound:
        return 0
    else:
        return (value - lower_bound) / (higher_bound - lower_bound) * 100


def calculate_agg_score(df):
    df_info = get_info()

    # Create a mapping from base task to full task (with fewshot)
    df["task_base"] = df["task"].str.rsplit("|", n=1).str[0]
    task_fewshot_mapping = df[["task_base", "task"]].drop_duplicates()
    df_info_with_fewshot = (
        df_info.merge(
            task_fewshot_mapping, left_on="task", right_on="task_base", how="left"
        )
        .drop(columns=["task_x", "task_base"])
        .rename(columns={"task_y": "task"})
    )

    df_no_max_samples = df.copy()
    df_no_max_samples["max_samples"] = "None"

    all_results = []  # List to collect DataFrames
    for (expe_name, tokens, FLOPs, max_samples), df_group in df_no_max_samples.groupby(
        ["expe_name", "tokens", "FLOPs", "max_samples"]
    ):
        df_group = df_info_with_fewshot.merge(
            df_group, on=["task", "metric"], how="left"
        )
        df_group = df_group.dropna(subset=["score"])
        df_group["norm_score"] = df_group.apply(
            lambda x: normalize_within_range(x["score"], x["random"], 1.0), axis=1
        )
        # Group by task type
        results_task = (
            df_group.groupby(["language", "task_type"])
            .agg({"norm_score": lambda x: x.mean(skipna=False)})
            .reset_index()
        )
        # Group by language
        results_final = (
            results_task.groupby("language")
            .agg({"norm_score": lambda x: x.mean(skipna=False)})
            .reset_index()
        )

        # Reformat
        results_task["expe_name"] = expe_name
        results_task["tokens"] = tokens
        results_task["FLOPs"] = FLOPs
        results_task["max_samples"] = max_samples
        results_task["metric"] = "agg"
        results_task["task"] = (
            results_task["task_type"]  # .str.upper()
            + " ("
            + results_task["language"]  # .str.upper()
            + ")"
        )
        results_task = results_task.rename(columns={"norm_score": "score"})
        all_results.append(results_task)

        results_final["expe_name"] = expe_name
        results_final["tokens"] = tokens
        results_final["FLOPs"] = FLOPs
        results_final["max_samples"] = max_samples
        results_final["metric"] = "agg"
        results_final["task"] = (
            "All together (" + results_final["language"] + ")"
        )  # .str.upper()
        results_final = results_final.rename(columns={"norm_score": "score"})
        all_results.append(results_final)

    if len(all_results) == 0:
        print("No results found for the given experiments.")
        return pd.DataFrame(
            columns=[
                "expe_name",
                "tokens",
                "FLOPs",
                "task",
                "max_samples",
                "metric",
                "score",
            ]
        )

    df = pd.concat(all_results, ignore_index=True)
    return df[
        ["expe_name", "tokens", "FLOPs", "task", "max_samples", "metric", "score"]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        nargs="+",
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )
    args = parser.parse_args()

    df = pd.concat([read_experiment_results(path) for path in args.experiment_path])
    df_agg = calculate_agg_score(df)
    print(df_agg)
