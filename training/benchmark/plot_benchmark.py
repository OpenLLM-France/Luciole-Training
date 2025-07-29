import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mtick


def create_config_name(entry):
    parts = []

    if "arch" in entry:
        parts.append(f"{entry['arch']}")

    if "tp" in entry:
        parts.append(f"tp{int(entry['tp'])}")

    if "pp" in entry:
        parts.append(f"pp{int(entry['pp'])}")

    if "cp" in entry:
        parts.append(f"cp{int(entry['cp'])}")

    if "precision" in entry:
        parts.append(entry["precision"])

    if "seq_length" in entry:
        parts.append(f"seq_length{int(entry['seq_length'])}")

    if "batch_size" in entry:
        parts.append(f"batch_size{int(entry['batch_size'])}")

    if entry.get("sequence_parallel", False):
        parts.append("sequence_parallel")

    return "_".join(parts)


def plot_training_and_gpu_hours(
    df, plot_name="plot.png", plot_title="", output_folder="", only_last_text=True
):
    # Find columns with only one unique value
    if len(df) == 0:
        print(f"Dataframe empty for {os.path.join(output_folder, plot_name)} ")
        return

    constant_columns = df.loc[:, df.nunique() == 1]

    # Log the dropped columns and their constant values
    dropped_info = {
        col: constant_columns[col].iloc[0] for col in constant_columns.columns
    }
    dropped_info.pop("num_nodes", None)
    dropped_info.pop("num_gpus", None)

    # Drop those columns from the DataFrame
    df = df.drop(columns=constant_columns.columns)

    # Create config names
    df["config"] = df.apply(create_config_name, axis=1)

    # Check for duplicate (config, num_nodes) pairs
    duplicates = df[df.duplicated(subset=["config", "num_nodes"], keep=False)]
    if not duplicates.empty:
        print("\n⚠️  Warning: The following config values are duplicated:")
        print(duplicates[["config", "num_nodes"]])
        print(df)
    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    metrics = [
        ("training_time", "Training Time for 1T Tokens", "Training time (in days)"),
        (
            "consumed_gpu_hours",
            "Consumed GPU Hours for 1T Tokens",
            "Consumed GPU hours (in thousand)",
        ),
    ]

    handles = []
    labels = []

    for ax, (y, title, ylabel) in zip(axes, metrics):
        sns.lineplot(data=df, x="num_gpus", y=y, hue="config", marker="o", ax=ax)

        for config_name, group in df.groupby("config"):
            if only_last_text:
                last_point = group.sort_values("num_gpus").iloc[-1]
                ax.text(
                    last_point["num_gpus"] + 3,
                    last_point[y],
                    f"{last_point[y]:.2f}"
                    if last_point[y] < 1000
                    else f"{last_point[y]/1000:.0f}k",
                    fontsize=10,
                    va="center",
                    ha="left",
                )
            else:
                last_points = group.sort_values("num_gpus")
                for _, point in last_points.iterrows():
                    ax.text(
                        point["num_gpus"] + 5,
                        point[y],
                        f"{point[y]:.2f}"
                        if point[y] < 1000
                        else f"{point[y]/1000:.0f}k",
                        fontsize=10,
                        va="center",
                        ha="left",
                    )

        ax.set_title(title)
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0)
        ax.grid(True)

        if y == "consumed_gpu_hours":
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:g}"
                )
            )

        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()

    # Add shared legend
    fig.legend(
        handles,
        labels,
        title="",
        loc="upper left",
        bbox_to_anchor=(0.92, 0.6),
        borderaxespad=0.0,
        fontsize="medium",
        title_fontsize="large",
    )

    # Add dropped info as annotation near the legend
    if dropped_info:
        info_str = "\n".join(f"- {k}: {v}" for k, v in dropped_info.items())
        legend_lines = len(labels)
        legend_height = 0.033 * legend_lines  # adjust this scaling if needed
        annotation_y = max(0.6 - legend_height - 0.05, 0.05)
        fig.text(
            0.92,
            annotation_y,
            f"Constant params:\n{info_str}",
            ha="left",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )

    # Final layout & save
    fig.suptitle(plot_title, fontsize=16)
    if plot_name:
        output_path = f"{output_folder}/{plot_name}" if output_folder else plot_name
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def load_data(input_folder):
    data = []
    folders = os.listdir(input_folder)

    for folder in folders:
        xp_folder = os.path.join(input_folder, folder)
        if not os.path.isfile(xp_folder):
            try:
                job_folder = [f for f in os.listdir(xp_folder) if f.startswith("job_")][
                    0
                ]
                job_folder = os.path.join(xp_folder, job_folder)
                files = os.listdir(job_folder)
                stat_file = [
                    f for f in files if f.startswith("stats_") and f.endswith(".json")
                ][0]
                stat_file = os.path.join(job_folder, stat_file)
                config_file = [
                    f for f in files if f.startswith("config_") and f.endswith(".json")
                ][0]
                config_file = os.path.join(job_folder, config_file)
                with open(stat_file, "r") as f:
                    stat_data = json.load(f)
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Error loading data for {folder}: {e}")
                continue
            if "args" not in config:
                config["args"] = {}
                config["args"]["arch"] = config["log"]["name"].split("_")[0]
                if isinstance(config["trainer"]["plugins"], list):
                    config["trainer"]["plugins"] = config["trainer"]["plugins"][0]
                config["args"]["fp8"] = "fp8" in config["trainer"]["plugins"]
            data.append(dict(**stat_data, **config))
    return data


def model_to_size(model_name):
    if model_name == "llama8b":
        return 7.5
    elif model_name == "llama1b":
        return 1.1
    elif model_name == "llama70b":
        return 988 * 1
    elif model_name == "mambahybrid8b":
        return 7.3


def convert_data(data):
    records = []
    for entry in data:
        try:
            batch = "global_batch_size"
            number_of_steps_per_trillion_tokens = 1e12 / (
                entry["data"][batch] * entry["data"]["seq_length"]
            )
            records.append(
                {
                    "num_nodes": entry["trainer"]["num_nodes"],
                    "num_gpus": entry["trainer"]["num_nodes"]
                    * entry["trainer"]["devices"],
                    "mean_step_timing": entry["mean_step_timings"],
                    "training_time": entry["mean_step_timings"]
                    * number_of_steps_per_trillion_tokens
                    / (3600 * 24),
                    "consumed_gpu_hours": (
                        entry["mean_step_timings"]
                        * number_of_steps_per_trillion_tokens
                        * entry["trainer"]["num_nodes"]
                        * entry["trainer"]["devices"]
                        / 3600
                    ),
                    "arch": entry["args"]["arch"],
                    "tp": entry["trainer"]["strategy"]["tensor_model_parallel_size"],
                    "pp": entry["trainer"]["strategy"]["pipeline_model_parallel_size"],
                    "precision": "fp8" if entry["args"]["fp8"] else "bf16",
                    "seq_length": entry["data"]["seq_length"],
                    "cp": entry["trainer"]["strategy"]["context_parallel_size"],
                    "batch_size": entry["data"][batch],
                    "sequence_parallel": entry["trainer"]["strategy"][
                        "sequence_parallel"
                    ],
                    "note": "\n" + entry.get("info", ""),
                    # "model_size": entry.get("model_size", model_to_size(entry["arch"])),
                }
            )
        except Exception as e:
            raise RuntimeError(f"error on {entry}") from e

    df = pd.DataFrame(records)
    return df


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--output_folder", default="plots")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    data = load_data(input_folder)

    df = convert_data(data)
    return output_folder, df, input_folder


if __name__ == "__main__":
    output_folder, df, input = setup()
    os.makedirs(output_folder, exist_ok=True)
    df = df.sort_values(by=["arch", "seq_length", "batch_size", "precision"])
    plot_training_and_gpu_hours(
        df,
        plot_name="all.png",
        plot_title="",
        output_folder=output_folder,
        only_last_text=False,
    )
