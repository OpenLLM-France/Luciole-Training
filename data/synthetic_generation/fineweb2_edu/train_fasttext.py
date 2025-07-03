from datasets import load_from_disk, load_dataset
import os
from fasttext import train_supervised
from utils import extract_educational_json, extract_text, normalize_text
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def print_results(model, valid_data, output_file):
    valid_texts = []
    true_labels = []
    with open(valid_data, "r") as f:
        for line in f:
            parts = line.strip().split()
            true_labels.append(parts[0])  # __label__xyz
            valid_texts.append(" ".join(parts[1:]))

    # Step 3: Predict labels for validation data
    predicted_labels = [model.predict(text)[0][0] for text in valid_texts]

    # Step 4: Confusion matrix
    labels = sorted(set(true_labels + predicted_labels))

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    short_labels = [label.replace("__label__", "")[:7] for label in labels]
    cm_df = pd.DataFrame(cm, index=short_labels, columns=short_labels)

    with open(output_file, "w") as f:
        print("\nConfusion Matrix:", file=f)
        print(cm_df.to_string(), file=f)

        # Step 5: Classification report
        print("\nClassification Report:", file=f)
        print(
            classification_report(
                true_labels, predicted_labels, labels=labels, zero_division=0
            ),
            file=f,
        )
    return


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT", "")

    parser = argparse.ArgumentParser(
        description="Train a fastText classifier on educational data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(
            main_path,
            "data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_fra_Latn_400k_2025-05-21T10-04-59.003164/default",
        ),
        help="Path to the input dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(main_path, "fasttext_classifiers/fineweb_fra_Latn"),
        help="Path to save the output model and data.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="educational_score",
        choices=["educational_score", "is_toxic", "is_ad", "topic"],
        help="Label to use for classification.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="softmax",
        choices=["softmax", "hs", "ova"],
        help="Loss function to use.",
    )
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument(
        "--use_normalize_text",
        action="store_true",
        help="Whether to normalize the text.",
    )
    parser.add_argument(
        "--from_parquet", action="store_true", help="read from parquet files."
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Add a quantize version of the model."
    )
    parser.add_argument(
        "--force", action="store_true", help="Whether to force overwrite."
    )
    args = parser.parse_args()

    epoch = args.epoch
    lr = args.lr
    ngrams = args.ngrams
    output_path = args.output_path
    input_path = args.input_path
    label = args.label
    use_normalize_text = args.use_normalize_text
    from_parquet = args.from_parquet
    loss = args.loss

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "model"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "data"), exist_ok=True)

    train_data = os.path.join(output_path, f"data/train_{label}.txt")
    valid_data = os.path.join(output_path, f"data/valid_{label}.txt")

    if os.path.exists(train_data) and os.path.exists(valid_data) and not args.force:
        print(f"The files {train_data} and {valid_data} already exist.")
    else:
        print(f"The files {train_data} and {valid_data} do not exist.")
        # Preprocess dataset
        if args.from_parquet:
            ds = load_dataset(
                "parquet", data_files={"train": os.path.join(input_path, "*.parquet")}
            )["train"]
        else:
            ds = load_from_disk(input_path)["train"]

        ds = ds.map(lambda x: extract_educational_json(x["generation"]))
        if "text" not in ds.column_names:
            ds = ds.map(lambda x: {"text": extract_text(x["instruction"])})
        ds = ds.map(
            lambda x: {"text": x["text"][:2000].replace("\n", " ") if x["text"] else ""}
        )
        if use_normalize_text:
            ds = ds.map(lambda x: {"text": normalize_text(x["text"])})
        ds = ds.filter(lambda x: x[label] is not None)
        print(ds[0])
        ds = ds.train_test_split(test_size=10000, seed=42, shuffle=True)

        # Write txt file
        with open(train_data, "w") as f:
            for sample in ds["train"]:
                f.write(
                    f"__label__{str(sample[label]).replace(' ', '_').lower()} {sample['text']}\n"
                )

        with open(valid_data, "w") as f:
            for sample in ds["test"]:
                f.write(
                    f"__label__{str(sample[label]).replace(' ', '_').lower()} {sample['text']}\n"
                )

    model_name = f"{label}_ngram{ngrams}_epoch{epoch}_lr{lr}"

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data,
        wordNgrams=ngrams,
        verbose=2,
        minCount=1,
        thread=40,
        epoch=epoch,
        lr=lr,
        loss=loss,
    )

    model.save_model(os.path.join(output_path, "model", f"{model_name}.bin"))

    print_results(
        model, valid_data, os.path.join(output_path, "model", f"{model_name}_logs.txt")
    )

    if args.quantize:
        model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
        print_results(*model.test(valid_data))
        model.save_model(os.path.join(output_path, "model", f"{model_name}.ftz"))
