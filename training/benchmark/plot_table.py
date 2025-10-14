import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
from plot_benchmark import setup


def summarize_training_times_by_arch_and_precision(df, show_fp8_only_if_better=False):
    if show_fp8_only_if_better:
        bf16_df = df[df["precision"] == "bf16"]
        fp8_df = df[df["precision"] == "fp8"]

        # {arch: (training_time, consumed_gpu_hours)} pour bf16
        bf16_stats = bf16_df.set_index("arch")[["training_time", "consumed_gpu_hours"]]

        def is_better_than_bf16(row):
            arch = row["arch"]
            if arch not in bf16_stats.index:
                return True  # pas de comparaison possible
            bf16_time, bf16_gpu = bf16_stats.loc[arch]
            return (
                row["training_time"] < bf16_time
                and row["consumed_gpu_hours"] < bf16_gpu
            )

        fp8_df_filtered = fp8_df[fp8_df.apply(is_better_than_bf16, axis=1)]

        df = pd.concat([bf16_df, fp8_df_filtered], ignore_index=True)
    grouped = df.copy()

    def format_cell(days, gpuh, error=None):
        if error and isinstance(error, str):
            return error
        return f"{days:.1f} days / {gpuh / 1000:.0f}k GPUh"

    summary = pd.DataFrame()
    summary["Architecture"] = df.apply(
        lambda row: f"{row['arch'].upper()}\n{row['fp8_recipe']}, grad_reduce: {row['grad_reduce_in_fp32']}\n({row['batch_size']}x{row['seq_length']} mb{row['micro_batch_size']} tp{row['tp']} pp{row['pp']} cp{row['cp']}){row['note']}",
        axis=1,
    )

    summary["training_time"] = grouped["training_time"]
    summary["1T tokens"] = [
        format_cell(t, g, t)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["3T tokens"] = [
        format_cell(t * 3, g * 3, t)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["5T tokens"] = [
        format_cell(t * 5, g * 5, t)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["10T tokens"] = [
        format_cell(t * 10, g * 10, t)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]

    summary["training_time"] = pd.to_numeric(summary["training_time"], errors="coerce")
    summary = summary.sort_values(by=["training_time"])
    summary = summary.drop(columns=["training_time"])
    return summary


def export_summary_to_markdown(df):
    return df.to_markdown(index=False)


def plot_training_time(summary_df, output_folder, plot_name=None):
    import re

    fig_height = len(summary_df) * 0.6 + 1  # adjust height based on number of rows
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.axis("tight")

    column_labels = summary_df.columns.tolist()
    table_data = summary_df.values.tolist()

    def extract_gpu_hours(cell):
        match = re.search(r"/\s*(\d+)k\s*GPUh", cell)
        return int(match.group(1)) if match else "error"

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_yellow_orange_red", ["#b6fcb6", "#fff89e", "#ffc074", "#ff8080"]
    )
    max_gpuh = 900

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            if j == 0:
                continue  # skip arch name
            gpuh = extract_gpu_hours(cell)
            if gpuh == "error":
                table[(i + 1, j)].set_facecolor("#777777")
            else:
                ratio = min(gpuh / max_gpuh, 1.0)
                color = cmap(ratio)
                table[(i + 1, j)].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_folder, plot_name + ".png" if plot_name else "plot_table.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


if __name__ == "__main__":
    output_folder, df, input = setup()
    print(df)
    os.makedirs(output_folder, exist_ok=True)
    table = summarize_training_times_by_arch_and_precision(df)
    print(table)
    markdown_table = export_summary_to_markdown(table)
    plot_training_time(table, output_folder, os.path.basename(input).split("_")[-1])
