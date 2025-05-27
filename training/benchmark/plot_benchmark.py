import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mtick


def plot_training_and_gpu_hours(
    df, plot_name="plot.png", plot_title="", output_folder=""
):
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
        sns.lineplot(data=df, x="num_nodes", y=y, hue="config", marker="o", ax=ax)

        for config_name, group in df.groupby("config"):
            last_point = group.sort_values("num_nodes").iloc[-1]
            ax.text(
                last_point["num_nodes"] + 0.5,
                last_point[y],
                f"{last_point[y]:.2f}"
                if last_point[y] < 1000
                else f"{last_point[y]/1000:.0f}k",
                fontsize=10,
                va="center",
                ha="left",
            )

        ax.set_title(title)
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0)
        ax.grid(True)

        if y == "consumed_gpu_hours":
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:g}"
                )
            )

        # Get legend handles/labels only once
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()  # Remove individual legends

    # Add shared legend to the right
    fig.legend(
        handles,
        labels,
        title="",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        borderaxespad=0.0,
        fontsize="medium",
        title_fontsize="large",
    )

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
    # plt.tight_layout()
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
        config_key = f"{entry['arch']}_tp{entry['tensor_parallelism']}_pp{entry['pipeline_parallelism']}_fp8{entry['fp8']}_seq_length{entry['seq_length']}"
        if "context_parallelism" in entry:
            config_key += f"_cp{entry['context_parallelism']}"
        else:
            config_key += "_cp1"
        records.append(
            {
                "num_nodes": entry["num_nodes"],
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
                "tp": entry["tensor_parallelism"],
                "pp": entry["pipeline_parallelism"],
                "fp8": entry["fp8"],
                "seq_length": entry["seq_length"],
                "cp": entry.get(
                    "context_parallelism", 1
                ),  # default to 1 if not present
                "config": config_key,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values(by="config")
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
        & (df["fp8"] == False)  # noqa E712
        & (df["seq_length"] == 4096)
        & (df["cp"] == 1)
    ]
    plot_training_and_gpu_hours(
        tp_df,
        plot_name="tp.png",
        plot_title="Impact of tensor parallelism on training",
        output_folder=output_folder,
    )

    arch_df = df[(df["arch"] == "llama8b") & (df["tp"] == 1) & (df["fp8"] == False)]  # noqa E712
    plot_training_and_gpu_hours(
        arch_df,
        plot_name="seq_length.png",
        plot_title="Impact of Seq Length on training",
        output_folder=output_folder,
    )

    arch_df = df[(df["pp"] == 1) & (df["tp"] == 1) & (df["cp"] == 1)]
    plot_training_and_gpu_hours(
        arch_df,
        plot_name="arch_fp8.png",
        plot_title="Impact of Architecture and FP8 on training",
        output_folder=output_folder,
    )
