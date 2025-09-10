from pathlib import Path
import json
import pandas as pd
import re
import numpy as np


def read_json_file(file_path, seq_length=2048):
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
    df["max_samples"] = str(data["config_general"]["max_samples"])
    df["tokens"] = (
        (df["model_name"].str.extract(r"-consumed_samples_([0-9.]+)")[0].astype(float))
        * seq_length
        / 10**9
    )
    df["steps"] = df["model_name"].str.extract(r"-step_([0-9.]+)")[0].astype(float)
    match = re.match(r"results_(.*)\.json", file_path.name)
    timestamp = match.group(1) if match else None
    df["timestamp"] = timestamp

    # Reorder columns
    df = df[["steps", "tokens", "timestamp", "task", "max_samples", "metric", "score"]]
    return df


def read_experiment_results(main_dir, seq_length=2048):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)

    json_files = list(main_dir.rglob("results_*.json"))
    if not json_files:
        print(f"No JSON result files found in {main_dir}")
        return

    # Read files and store relative path + DataFrame
    dataframes = []
    for file in json_files:
        file_path = Path(file)
        parts = file_path.parts
        try:
            eval_index = parts.index("evaluation")
            expe_name = parts[
                eval_index - 1
            ]  # Name of the folder just before 'evaluation'
        except ValueError:
            expe_name = "unknown"

        df = read_json_file(file, seq_length=seq_length)
        df["expe_name"] = expe_name
        dataframes.append(df)

    # Concatenate all DataFrames
    df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H-%M-%S.%f")

    # Keep the most recent row for each (model_name, tokens, task, metric) tuple
    df_latest = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["expe_name", "steps", "tokens", "task", "max_samples", "metric"],
            keep="last",
        )
        .drop("timestamp", axis=1)
    )
    return df_latest[
        ["expe_name", "steps", "tokens", "task", "max_samples", "metric", "score"]
    ]


def read_datamix(main_dir):
    json_files = list(Path(main_dir).glob("datamix/*.json"))
    if not json_files:
        print(f"No JSON datamix file found in {main_dir}")
        return
    if len(json_files) > 1:
        print(f"More than one JSON datamix file found in {main_dir}")
        return
    json_file = json_files[0]
    with open(json_file, "r") as f:
        data = json.load(f)
    datamix = data["train"]
    return datamix


# Function to fit regression for each group
def compute_regression(group):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    group = group[group["tokens"] > 0]  # adjust to your x column name

    group = group.sort_values("tokens")
    x = group["tokens"].values.reshape(-1, 1)
    y = group["score"].values

    tokens = x.flatten().tolist()
    score = y.tolist()

    if len(group) < 2:
        return pd.DataFrame(
            [{"slope": np.nan, "intercept": np.nan, "tokens": tokens, "score": score}]
        )

    model = LinearRegression()
    model.fit(np.log(x), y)
    y_pred = model.predict(np.log(x))

    return pd.DataFrame(
        [
            {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r2": r2_score(y, y_pred),
                "tokens": tokens,
                "score": score,
            }
        ]
    )


def process_results(df):
    # Groupby expe, task and metric and fit regression
    group_df = (
        df.groupby(["task", "max_samples", "metric", "expe_name"])
        .apply(compute_regression)
        .reset_index()
    )
    return group_df
