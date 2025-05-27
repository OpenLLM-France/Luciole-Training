from datasets import load_from_disk, load_dataset
import os
from fasttext import train_supervised
from utils import extract_educational_json, extract_text, normalize_text
import argparse


def print_results(N, p, r):
    print("-" * 7)
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    print("-" * 7)


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
            "synthetic_data/fra_Latn_data/Qwen3-32B_multi_task_2025-05-16T16-26-25.365295",
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
    args = parser.parse_args()

    epoch = args.epoch
    lr = args.lr
    ngrams = args.ngrams
    output_path = args.output_path
    input_path = args.input_path
    label = args.label
    use_normalize_text = args.use_normalize_text
    from_parquet = args.from_parquet

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "model"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "data"), exist_ok=True)

    train_data = os.path.join(output_path, f"data/train_{label}.txt")
    valid_data = os.path.join(output_path, f"data/valid_{label}.txt")

    if os.path.exists(train_data) and os.path.exists(valid_data):
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
        ds = ds.train_test_split(test_size=1000, seed=42, shuffle=True)

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
    )

    print_results(*model.test(valid_data))
    model.save_model(os.path.join(output_path, "model", f"{model_name}.bin"))

    if args.quantize:
        model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
        print_results(*model.test(valid_data))
        model.save_model(os.path.join(output_path, "model", f"{model_name}.ftz"))
