import os
from distilabel.distiset import Distiset
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import json

def extract_educational_json(text: str) -> dict | None:
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)

    matches = pattern.findall(text)
    match = matches[0] 
    try:
        data_dict = json.loads(match)
        return data_dict
    except json.JSONDecodeError:
        return None

def plot_confusion_matrix(ds, expe_name, output_dir="out"):
    # Assuming `ds['extracted_score']` and `ds['score']` are arrays or pandas Series
    y_pred = ds['educational_score']
    y_true = ds['score']

    # Define all possible labels from -1 to 5
    labels = list(range(-1, 6))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix: {expe_name}")
    plt.xlabel("Qwen Score")
    plt.ylabel("Finweb Score")
    plt.grid(False)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"confusion_{expe_name.replace('/', '-')}.png"))

def plot_histograms(ds, expe_name, output_dir="out"):
    # Extract columns
    educational_scores = ds['educational_score']
    harmfulness_scores = ds['toxicity_score']
    topics = ds['topic']

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Histogram for educational_score
    axs[0].hist(educational_scores, bins=6, range=(0, 6), alpha=0.7, align='left', rwidth=0.8)
    axs[0].set_title('Educational Score')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Frequency')

    # Histogram for harmfulness_score
    axs[1].hist(harmfulness_scores, bins=6, range=(0, 6), alpha=0.7, align='left', rwidth=0.8)
    axs[1].set_title('Toxicity Score')
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Frequency')

    # Histogram for topic (categorical) — count frequencies
    topic_counts = Counter(topics)
    axs[2].bar(topic_counts.keys(), topic_counts.values(), alpha=0.7)
    axs[2].set_title('Topic Distribution')
    axs[2].set_xlabel('Topic')
    axs[2].set_ylabel('Frequency')
    axs[2].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"hist_{expe_name.replace('/', '-')}.png"))

if __name__ == "__main__":

    for expe_name in [
        # "Qwen3-14B_en_2025-05-15T18-05-12.214286",
        # "Qwen3-8B_en_2025-05-15T17-48-58.241512",
        # "Qwen3-8B_scale_2025-05-15T18-17-32.125437",
        # "Qwen3-14B_scale_2025-05-15T18-25-29.448993",
        # "Qwen3-32B_scale_2025-05-15T18-30-45.785649",
        # "Qwen3-32B_en_2025-05-16T10-05-48.950671",
        # "Qwen3-32B_multi_task_2025-05-16T11-25-11.332632", 
        # "Qwen3-32B_multi_task_2025-05-16T14-14-44.721269", 
        # "en_data/Qwen3-32B_multi_task_2025-05-16T15-44-06.564778"
        "fra_Latn_data/Qwen3-32B_multi_task_2025-05-16T16-21-09.458535"
    ]:
        expe_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "synthetic_data/", expe_name)
        distiset = Distiset.load_from_disk(expe_path)
        ds = distiset["default"]["train"]
        ds = ds.map(lambda x: extract_educational_json(x["generation"]))
        # print(ds)

        # plot_confusion_matrix(ds, expe_name, output_dir="out")
        plot_histograms(ds, expe_name, output_dir="out")

        # print("\nList of subtopics:")
        # print(list(set(ds['subtopic'])))

        # ds = ds.filter(lambda x: x["educational_score"] >= 4) 
        # print(ds[0]['instruction'])
        # print(ds[0]['generation'])

