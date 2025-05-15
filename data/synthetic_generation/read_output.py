import os
from datasets import load_dataset
from distilabel.distiset import Distiset
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import warnings

def plot_confusion_matrix(df, name_1, name_2, filename="confusion_matrix.png"):
    scores1 = df[df['name'] == name_1]['scores'].iloc[0]
    scores2 = df[df['name'] == name_2]['scores'].iloc[0]

    labels = list(range(-1, 6))  # Assuming scores range from -1 to 5

    cm = confusion_matrix(scores1, scores2, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.xlabel(name_2)
    plt.ylabel(name_1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_histograms(df):
    # Define bounds
    x_min, x_max = -1, 5
    bins = range(x_min, x_max + 2)  # +2 because of upper-exclusive

    # Plot
    fig, axes = plt.subplots(len(df), 1, figsize=(8, 3 * len(df)), sharex=True)

    if len(df) == 1:
        axes = [axes]

    for i, (dataset, scores) in enumerate(zip(df["name"], df["scores"])):
        axes[i].hist(scores, bins=bins, align='left', rwidth=0.8)
        axes[i].set_title(dataset)
        axes[i].set_ylabel("Frequency")
        axes[i].set_xlim(x_min - 0.5, x_max + 0.5)
        axes[i].set_xticks(range(x_min, x_max + 1))

    axes[-1].set_xlabel("Score")
    plt.tight_layout()

    plt.savefig("histograms.png")

# Clip and warn if needed
def clip_and_warn(scores):
    x_min, x_max = -1, 5
    clipped = []
    for s in scores:
        if s < x_min or s > x_max:
            warnings.warn(f"Score {s} out of bounds; clipping to [{x_min}, {x_max}]")
            s = min(max(s, x_min), x_max)
        clipped.append(s)
    return clipped

main_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "synthetic_data") 

out = []

for dataset_name in os.listdir(main_path):
    # Load dataset from disk using Distiset
    expe_path = os.path.join(main_path, dataset_name)
    distiset = Distiset.load_from_disk(expe_path)
    ds = distiset["default"]["train"]

    # Map over the dataset to extract the score
    def extract_score(example):
        match = re.search(
            r"(?:Educational score:|Note éducative\s*:)\s*\**(\d)\s*(?:points)?\**",
            example["generation"],
            re.IGNORECASE
        )
        if match:
            return {"score": int(match.group(1))}
        else:
            return {"score": -1}  # or None if you prefer

    ds = ds.map(extract_score)

    # print(ds["score"])
    print(dataset_name)
    print(ds[-1]['generation'])
    print('<<<<<<<<<\n')
    out.append({"name": dataset_name, "scores": ds["score"]})

df = pd.DataFrame(out)
df["scores"] = df["scores"].apply(clip_and_warn)
df.sort_values(by="name", inplace=True)
df[['model', 'language', 'date']] = df['name'].str.extract(
    r'^(?P<model>.+)_(?P<language>fr|en)_(?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)$'
)
print(df)

plot_histograms(df)
plot_confusion_matrix(df, "Qwen3-32B_en_2025-05-15T16-14-58.654993", "Qwen3-8B_en_2025-05-15T15-37-29.144996", "confusion_qwen.png")
plot_confusion_matrix(df, "Llama-3.1-8B-Instruct_en_2025-05-15T16-04-47.059051", "Qwen3-8B_en_2025-05-15T15-37-29.144996", "confusion_llama_qwen.png")
plot_confusion_matrix(df, "Qwen3-8B_en_2025-05-15T15-37-29.144996", "Qwen3-8B_fr_2025-05-15T15-41-13.785139", "confusion_qwen_8B_language.png")
plot_confusion_matrix(df, "Qwen3-32B_en_2025-05-15T16-14-58.654993", "Qwen3-32B_fr_2025-05-15T16-57-11.168984", "confusion_qwen_32B_language.png")
