from utils import read_experiment_results
import pandas as pd
import argparse


def read_info():
    df_info = pd.read_json("agg_tasks.jsonl", lines=True)
    df_info["random"] = (1.0 / df_info["num_classes"]).fillna(0.0)
    return df_info


def normalize_within_range(value, lower_bound, higher_bound):
    if value < lower_bound:
        return 0
    else:
        return (value - lower_bound) / (higher_bound - lower_bound) * 100


def calculate_agg_score(df):
    df_info = read_info()

    all_results = []  # List to collect DataFrames
    for (expe_name, tokens, max_samples), df_group in df.groupby(
        ["expe_name", "tokens", "max_samples"]
    ):
        df_group = df_info.merge(df_group, on=["task", "metric"], how="left")
        df_group["norm_score"] = df_group.apply(
            lambda x: normalize_within_range(x["score"], x["random"], 1.0), axis=1
        )
        # Group by task type
        results_task = (
            df_group.groupby(["language", "task_type", "random"])
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
        results_task["max_samples"] = max_samples
        results_task["metric"] = "agg"
        results_task["task"] = (
            "AGG_"
            + results_task["language"].str.upper()
            + "_"
            + results_task["task_type"].str.upper()
        )
        results_task = results_task.rename(columns={"norm_score": "score"})
        all_results.append(results_task)

        results_final["expe_name"] = expe_name
        results_final["tokens"] = tokens
        results_final["max_samples"] = max_samples
        results_final["metric"] = "agg"
        results_final["task"] = "AGG_" + results_final["language"].str.upper()
        results_final = results_final.rename(columns={"norm_score": "score"})
        all_results.append(results_final)

    if len(all_results) == 0:
        print("No results found for the given experiments.")
        return pd.DataFrame(
            columns=[
                "expe_name",
                "tokens",
                "task",
                "max_samples",
                "metric",
                "random",
                "score",
            ]
        )

    df = pd.concat(all_results, ignore_index=True)
    return df[
        ["expe_name", "tokens", "task", "max_samples", "metric", "random", "score"]
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
