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

    return "_".join(parts)


def plot_training_and_gpu_hours(
    df, plot_name="plot.png", plot_title="", output_folder=""
):
    # Find columns with only one unique value
    constant_columns = df.loc[:, df.nunique() == 1]

    # Log the dropped columns and their constant values
    dropped_info = {
        col: constant_columns[col].iloc[0] for col in constant_columns.columns
    }
    print(dropped_info)

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
            last_point = group.sort_values("num_gpus").iloc[-1]
            ax.text(
                last_point["num_gpus"] + 0.5,
                last_point[y],
                f"{last_point[y]:.2f}"
                if last_point[y] < 1000
                else f"{last_point[y]/1000:.0f}k",
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
        bbox_to_anchor=(0.92, 0.5),
        borderaxespad=0.0,
        fontsize="medium",
        title_fontsize="large",
    )

    # Add dropped info as annotation near the legend
    if dropped_info:
        info_str = "\n".join(f"- {k}: {v}" for k, v in dropped_info.items())
        legend_lines = len(labels)
        legend_height = 0.03 * legend_lines  # adjust this scaling if needed
        annotation_y = max(0.5 - legend_height - 0.05, 0.05)
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
    output_path = f"{output_folder}/{plot_name}" if output_folder else plot_name
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def load_data(input_folder):
    data = []
    folders = os.listdir(input_folder)

    for folder in folders:
        xp_folder = os.path.join(input_folder, folder)
        if not os.path.isfile(xp_folder):
            files = os.listdir(xp_folder)
            stat_files = [
                f for f in files if f.startswith("stats_") and f.endswith(".json")
            ]
            if len(stat_files) > 0:
                stat_file = os.path.join(xp_folder, stat_files[0])
                with open(stat_file, "r") as f:
                    json_data = json.load(f)
                    data.append(json_data)
    return data


def convert_data(data):
    records = []
    for entry in data:
        number_of_steps_per_trillion_tokens = 1e12 / (
            entry["batch_size"] * entry["seq_length"]
        )
        records.append(
            {
                "num_nodes": entry["num_nodes"],
                "num_gpus": entry["num_nodes"] * 4,
                "mean_step_timing": entry["mean_step_timings"],
                "training_time": entry["mean_step_timings"]
                * number_of_steps_per_trillion_tokens
                / (3600 * 24),
                "consumed_gpu_hours": (
                    entry["mean_step_timings"]
                    * number_of_steps_per_trillion_tokens
                    * entry["num_nodes"]
                    * 4
                    / 3600
                ),
                "arch": entry["arch"],
                "tp": entry["tensor_model_parallel_size"],
                "pp": entry["pipeline_model_parallel_size"],
                "precision": "fp8" if entry["fp8"] else "bf16",
                "seq_length": entry["seq_length"],
                "cp": entry["context_parallel_size"],
                "batch_size": entry["batch_size"],
            }
        )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--output_folder", default="")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    data = load_data(input_folder)

    df = convert_data(data)

    plot_training_and_gpu_hours(
        df, plot_name="all.png", plot_title="", output_folder=output_folder
    )

    tp_df = df[
        (df["arch"] == "llama8b")
        & (df["pp"] == 1)
        & (df["precision"] == "bf16")  # noqa E712
        & (df["seq_length"] == 4096)
        & (df["cp"] == 1)
    ]
    tp_df = tp_df.sort_values(by=["tp", "batch_size"])
    plot_training_and_gpu_hours(
        tp_df,
        plot_name="tp.png",
        plot_title="Impact of tensor parallelism on training",
        output_folder=output_folder,
    )

    seq_df = df[
        (df["arch"] == "llama8b") & (df["tp"] == 1) & (df["precision"] == "bf16")
    ]  # noqa E712
    seq_df = seq_df.sort_values(by=["seq_length", "batch_size", "pp", "cp"])
    plot_training_and_gpu_hours(
        seq_df,
        plot_name="seq_length.png",
        plot_title="Impact of Seq Length on training",
        output_folder=output_folder,
    )

    arch_df = df[
        (df["pp"] == 1)
        & (df["tp"] == 1)
        & (df["cp"] == 1)
        & ((df["batch_size"] == 1024) | (df["arch"] == "llama1b"))
    ]
    arch_df = arch_df.sort_values(by=["arch", "precision"])
    plot_training_and_gpu_hours(
        arch_df,
        plot_name="arch_fp8.png",
        plot_title="Impact of Architecture and FP8 on training",
        output_folder=output_folder,
    )

    arch_df = df[
        (df["pp"] == 1) & (df["tp"] == 1) & (df["cp"] == 1) & (df["arch"] == "llama8b")
    ]
    arch_df = arch_df.sort_values(by=["precision", "batch_size"])
    plot_training_and_gpu_hours(
        arch_df,
        plot_name="batch_size_fp8.png",
        plot_title="Impact of Architecture and FP8 on training",
        output_folder=output_folder,
    )
