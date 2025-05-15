import os
from datasets import load_dataset
from distilabel.distiset import Distiset
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(output_1, output_2):
    scores1 = output_1['scores']
    scores2 = output_2['scores']

    labels = range(-1, 6)  # Assuming scores range from -1 to 5

    cm = confusion_matrix(scores1, scores2, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap="Blues")
    plt.xlabel(f"{output_2['dataset']}")
    plt.ylabel(f"{output_1['dataset']}")

    plt.savefig("confusion_matrix.png")

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
    print(ds[-1]['generation'])

    out.append({"dataset": dataset_name, "scores": ds["score"]})

print(out)

plot_confusion_matrix(out[0], out[1])

print(distiset)