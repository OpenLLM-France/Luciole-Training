import os
from distilabel.distiset import Distiset
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def extract_score(example):
    match = re.search(
        r"(?:Educational score:\s*\*{0,2})(\d)(?:\*{0,2})",
        example["generation"],
        re.IGNORECASE
    )
    if match:
        return {"extracted_score": int(match.group(1))}
    else:
        return {"extracted_score": -1}  # or None if you prefer

def plot_confusion_matrix(ds, expe_name, output_dir="out"):
    # Assuming `ds['extracted_score']` and `ds['score']` are arrays or pandas Series
    y_pred = ds['extracted_score']
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
    plt.savefig(os.path.join(output_dir, f"{expe_name}.png"))

if __name__ == "__main__":

    for expe_name in [
        "Qwen3-14B_en_2025-05-15T18-05-12.214286",
        "Qwen3-8B_en_2025-05-15T17-48-58.241512",
        "Qwen3-8B_scale_2025-05-15T18-17-32.125437",
        "Qwen3-14B_scale_2025-05-15T18-25-29.448993",
        "Qwen3-32B_scale_2025-05-15T18-30-45.785649",
        "Qwen3-32B_en_2025-05-16T10-05-48.950671"
    ]:
        expe_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "synthetic_data/en_data/", expe_name)
        distiset = Distiset.load_from_disk(expe_path)
        ds = distiset["default"]["train"]
        ds = ds.map(extract_score)

        plot_confusion_matrix(ds, expe_name, output_dir="out")
        print(f"Model: {expe_name}\n\nGeneration:")
        print(ds[:5]['generation'])
        print("\n----------\n")

