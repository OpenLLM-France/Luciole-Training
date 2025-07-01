import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
from plot_benchmark import setup


def summarize_training_times_by_arch_and_precision(df):
    bf16_df = df[df["precision"] == "bf16"]
    fp8_df = df[df["precision"] == "fp8"]

    # Créer un dictionnaire {arch: (training_time, consumed_gpu_hours)} pour bf16
    bf16_stats = bf16_df.set_index("arch")[["training_time", "consumed_gpu_hours"]]

    # Filtrer fp8 : ne garder que si meilleure que bf16
    def is_better_than_bf16(row):
        arch = row["arch"]
        if arch not in bf16_stats.index:
            return True  # pas de comparaison possible
        bf16_time, bf16_gpu = bf16_stats.loc[arch]
        return row["training_time"] < bf16_time and row["consumed_gpu_hours"] < bf16_gpu

    fp8_df_filtered = fp8_df[fp8_df.apply(is_better_than_bf16, axis=1)]

    # Combiner les deux
    df = pd.concat([bf16_df, fp8_df_filtered], ignore_index=True)

    grouped = (
        df.groupby(["arch", "precision"])[["training_time", "consumed_gpu_hours"]]
        .mean()
        .reset_index()
    )

    def format_cell(days, gpuh):
        return f"{days:.1f} days / {gpuh / 1000:.0f}k GPUh"

    summary = pd.DataFrame()
    summary["Architecture"] = grouped["arch"] + " (" + grouped["precision"] + ")"

    summary["1T tokens"] = [
        format_cell(t, g)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["3T tokens"] = [
        format_cell(t * 3, g * 3)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["5T tokens"] = [
        format_cell(t * 5, g * 5)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]
    summary["10T tokens"] = [
        format_cell(t * 10, g * 10)
        for t, g in zip(grouped["training_time"], grouped["consumed_gpu_hours"])
    ]

    # Définir l'ordre fixe
    custom_order = [
        "llama1b",
        "llama3b",
        "llama8b",
        "llama70b",
        "nemotronh8b",
        "nemotronh56b",
        "mixtral7x8",
    ]

    # Extraire arch pure pour trier
    summary["model_name"] = grouped["arch"]

    summary["Architecture"] = grouped["arch"] + " (" + grouped["precision"] + ")"
    summary["model_order"] = pd.Categorical(
        summary["model_name"], categories=custom_order, ordered=True
    )
    # Sort precision so fp8 comes before bf16
    precision_order = pd.Categorical(
        grouped["precision"], categories=["fp8", "bf16"], ordered=True
    )
    summary["precision_order"] = precision_order

    # Combine model + precision order
    summary = summary.sort_values(by=["model_order", "precision_order"])

    # Drop helper columns
    summary = summary.drop(columns=["model_name", "model_order", "precision_order"])

    return summary


def export_summary_to_markdown(df):
    return df.to_markdown(index=False)


def plot_training_time(summary_df, output_folder):
    import re

    fig_height = len(summary_df) * 0.6 + 1  # adjust height based on number of rows
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.axis("tight")

    column_labels = summary_df.columns.tolist()
    table_data = summary_df.values.tolist()

    # Fonction d'extraction des GPUh à partir de la chaîne "x.x days / yyk GPUh"
    def extract_gpu_hours(cell):
        match = re.search(r"/\s*(\d+)k\s*GPUh", cell)
        return int(match.group(1)) if match else 0

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_yellow_orange_red", ["#b6fcb6", "#fff89e", "#ffc074", "#ff8080"]
    )
    max_gpuh = 900  # 1000k GPUh correspond au rouge

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],  # force table to occupy full figure
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Appliquer la couleur aux cellules contenant du GPUh
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            if j == 0:
                continue  # skip arch name
            gpuh = extract_gpu_hours(cell)
            ratio = min(gpuh / max_gpuh, 1.0)
            color = cmap(ratio)
            table[(i + 1, j)].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "plot_table.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


if __name__ == "__main__":
    output_folder, df = setup()
    os.makedirs(output_folder, exist_ok=True)
    table = summarize_training_times_by_arch_and_precision(df)
    print(table)
    markdown_table = export_summary_to_markdown(table)
    plot_training_time(table, output_folder)
