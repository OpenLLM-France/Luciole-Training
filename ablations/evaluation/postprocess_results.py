import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import math

def read_file(file):
    with open(file, 'r') as file:
        data = json.load(file)
    model_name = data['config_general']['model_name']
    results = data["results"]
    df = pd.DataFrame.from_dict(results, orient='index').reset_index(names="task")
    df['model_name'] = model_name
    return df

def read_all_results(main_dir):
    main_dir = Path(main_dir)
    json_files = main_dir.rglob('*.json')  # recursively finds all .json files
    df = pd.concat([read_file(file) for file in json_files])
    df['datamix'] = df['model_name'].str.extract(r'huggingface_checkpoints_datamix_(.*?)_4n_20b--')
    df['step'] = df['model_name'].str.extract(r'--step_([0-9.]+)-')[0].astype(float)
    df['samples'] = df['model_name'].str.extract(r'-consumed_samples_([0-9.]+)')[0].astype(float)
    df['tokens'] = df['samples'] * 2048 / 10**9
    df['fr_prop'] = df['model_name'].str.extract(r'([0-9.]+)_fra_Latn')[0].astype(float)
    df['en_prop'] = df['model_name'].str.extract(r'([0-9.]+)_eng_Latn')[0].astype(float)
    return df

def plot_task(ax, df, task, metric, xlog=False):
    df = df[df['task'] == task]
    df = df.sort_values('tokens')

    pivot_df = df.pivot(index='tokens', columns='datamix', values=metric)
    stderr_df = df.pivot(index='tokens', columns='datamix', values=metric + '_stderr')

    for col in pivot_df.columns:
        mean = pivot_df[col].dropna()
        stderr = stderr_df[col].dropna()

        # Align indices in case they differ slightly after dropna
        common_index = mean.index.intersection(stderr.index)
        mean = mean.loc[common_index]
        stderr = stderr.loc[common_index]

        ax.plot(mean.index, mean.values, marker='+', label=col, alpha=0.8)
        ax.fill_between(mean.index, mean - stderr, mean + stderr, alpha=0.1)

    ax.set_xlabel('B tokens')
    ax.set_ylabel(metric)
    ax.set_title(task)
    if xlog:
        ax.set_xscale('log')


def plot_list_of_tasks(df, list_of_tasks_to_plot, output_dir, title=None, xlog=False):
    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    # Store handles and labels for the shared legend
    legend_handles, legend_labels = None, None

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(axes[i], df, task, metric, xlog=xlog)
        
        # Grab the legend handles and labels from the first plot (or any plot)
        if legend_handles is None:
            handles, labels = axes[i].get_legend_handles_labels()
            legend_handles, legend_labels = handles, labels

    # Dedicated subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        title="datamix",
        loc="center",
        # prop={'size': 14},        
        # title_fontsize=16        
    )

    # Hide any other unused subplots if any
    for j in range(len(list_of_tasks_to_plot), len(axes) - 1):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir)

if __name__=="__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")
    main_dir = os.path.join(main_path, "ablations/evaluation/language_ablations")
    df = read_all_results(main_dir)

    # English benchmarks
    list_of_tasks_to_plot = [
        ("helm|boolq|0", "pem"),
        ("lighteval|triviaqa|0", "qem"),
        ("lighteval|arc:easy|0", "acc"),
        ("lighteval|arc:easy|0", "acc_norm"),
        ("leaderboard|arc:challenge|0", "acc"),
        ("leaderboard|arc:challenge|0", "acc_norm"),
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|openbookqa|0", "acc_norm"),
        ("lighteval|piqa|0", "acc_norm")    
    ]
    output_dir = os.path.join(main_path, "ablations/evaluation/language_ablations/en.png")
    plot_list_of_tasks(df, list_of_tasks_to_plot, output_dir, title="English Tasks", xlog=False)

    # French benchmarks
    list_of_tasks_to_plot = [
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"), 
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"), 
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm"), 
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm"), 
        ("lighteval|xcodah_fra_cf|0", "acc_norm"), 
        ("lighteval|xcsqa_fra_cf|0", "acc_norm"),     
        ("lighteval|xnli2.0_fra_cf|0", "acc_norm"),     
        ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"), 
        ("lighteval|fquadv2_fra|0", "f1_fra"),     
        ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"), 
        ("lighteval|mintaka_fra|0", "f1_fra"), 
    ]
    output_dir = os.path.join(main_path, "ablations/evaluation/language_ablations/fr.png")
    plot_list_of_tasks(df, list_of_tasks_to_plot, output_dir, title="French Tasks", xlog=False)

    # French benchmarks
    list_of_tasks_to_plot = [
        ("lighteval|belebele_fra_Latn_cf|5", "acc_norm"), 
        ("lighteval|mlmm_arc_fra_cf:challenge|5", "acc_norm"), 
        ("lighteval|mlmm_hellaswag_fra_cf|5", "acc_norm"), 
        ("lighteval|xcodah_fra_cf|5", "acc_norm"), 
        ("lighteval|xcsqa_fra_cf|5", "acc_norm"),     
        ("lighteval|xnli2.0_fra_cf|5", "acc_norm"),     
        ("lighteval|fquadv2_fra|5", "exact_match_fra_prefix"), 
        ("lighteval|fquadv2_fra|5", "f1_fra"),     
        ("lighteval|mintaka_fra|5", "exact_match_fra_prefix"), 
        ("lighteval|mintaka_fra|5", "f1_fra"), 
    ]
    output_dir = os.path.join(main_path, "ablations/evaluation/language_ablations/fr_5_shots.png")
    plot_list_of_tasks(df, list_of_tasks_to_plot, output_dir, title="French Tasks (5-shots)", xlog=False)

    # French benchmarks
    main_dir = os.path.join(main_path, "ablations/evaluation/language_ablations/fr_cf")
    df = read_all_results(main_dir)
    list_of_tasks_to_plot = [
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm"), 
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm_pmi"), 
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm_token"), 
    ]
    output_dir = os.path.join(main_path, "ablations/evaluation/language_ablations/fr_cf.png")
    plot_list_of_tasks(df, list_of_tasks_to_plot, output_dir, title="French MMLU", xlog=False)