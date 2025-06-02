from pathlib import Path
import json
import pandas as pd
import re
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score


def read_json_file(file_path):
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)

    df = (
        pd.DataFrame(data["results"])
        .stack()
        .reset_index(name="score")
        .rename(columns={"level_0": "metric", "level_1": "task"})
    )

    # Filter out metrics ending in "_stderr"
    df = df[~df["metric"].str.endswith("_stderr")]

    # Add model name and timestamp
    df["model_name"] = data["config_general"]["model_name"]
    df["tokens"] = (
        (df["model_name"].str.extract(r"-consumed_samples_([0-9.]+)")[0].astype(float))
        * 2048
        / 10**9
    )

    match = re.match(r"results_(.*)\.json", file_path.name)
    timestamp = match.group(1) if match else None
    df["timestamp"] = timestamp

    # Reorder columns
    df = df[["tokens", "timestamp", "task", "metric", "score"]]
    return df


def read_experiment_results(main_dir):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)

    json_files = list(main_dir.rglob("results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {main_dir}")

    # Read files and store relative path + DataFrame
    dataframes = []
    for file in json_files:
        relative_path = file.relative_to(main_dir)
        # Extract portion before "/evaluation/"
        parts = relative_path.parts
        eval_index = parts.index("evaluation")
        expe_name = Path(*parts[:eval_index]).name
        df = read_json_file(file)
        df["expe_name"] = expe_name
        dataframes.append(df)

    # Concatenate all DataFrames
    df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H-%M-%S.%f")

    # Keep the most recent row for each (model_name, tokens, task, metric) tuple
    df_latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["expe_name", "tokens", "task", "metric"], keep="last")
        .drop("timestamp", axis=1)
    )
    return df_latest


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


# Function to fit regression for each group
def compute_regression(group):
    group = group[group["tokens"] > 0]  # adjust to your x column name

    if len(group) < 2:
        return pd.DataFrame(
            [{"slope": np.nan, "intercept": np.nan, "tokens": [], "score": []}]
        )

    group = group.sort_values("tokens")
    x = group["tokens"].values.reshape(-1, 1)
    y = group["score"].values

    model = LinearRegression()
    model.fit(np.log(x), y)
    y_pred = model.predict(np.log(x))

    return pd.DataFrame(
        [
            {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r2": r2_score(y, y_pred),
                "tokens": x.flatten().tolist(),
                "score": y.tolist(),
            }
        ]
    )


def process_results(path):
    df = read_experiment_results(path)
    group_df = (
        df.groupby(["task", "metric", "expe_name"])
        .apply(compute_regression)
        .reset_index()
    )
    return group_df
    # # Load task info mapping
    # task_info_mapping = read_info_baseline()
    # df_info = df['task'].apply(lambda t: get_info(t, task_info_mapping)).apply(pd.Series)
    # df = df.join(df_info)
    # # Normalize
    # df['norm_score'] = df.apply(lambda x: normalize_within_range(x['score'], x['random'], 1.), axis=1)
    # # Fit linear regression
    # return df
