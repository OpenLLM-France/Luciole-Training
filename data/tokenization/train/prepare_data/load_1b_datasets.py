from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter, LambdaFilter
from datatrove.pipeline.writers import ParquetWriter
from datatrove.data import DocumentsPipeline
import pandas as pd
import argparse
import os


def read_markdown_table(filepath="fineweb2_languages.md"):
    df = pd.read_csv(filepath, sep="|", engine="python", index_col=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [col.strip() for col in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Remove first and last rows
    df = df.iloc[1:-1]
    # Format columns
    df["Subset"] = df["Subset"].str.replace("`", "", regex=False)
    df["Words"] = df["Words"].str.replace(",", "").astype(int)
    df["Documents"] = df["Documents"].str.replace(",", "").astype(int)
    return df


def remove_metadata(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    """
    `data` is a generator of Document. You must also return a generator of Document (yield)
    You can optionally use `rank` and `world_size` for sharding
    """
    for doc in data:
        doc.metadata = {}
        yield doc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--language",
        type=str,
        default="fra_Latn",
        help="Language to process",
    )
    args = argparser.parse_args()

    main_path = os.getenv("OpenLLM_OUTPUT")
    language = args.language
    output_path = os.path.join(main_path, "data/data_for_tokenization")
    target_num_words = 1e9

    # Read stats from fineweb2
    if language in [
        "fra_Latn",
        "deu_Latn",
        "ita_Latn",
        "spa_Latn",
        "por_Latn",
        "nld_Latn",
        "arb_Arab",
    ]:
        df = read_markdown_table("fineweb2_languages.md")
        selected_row = df[df["Subset"] == language].iloc[0]
        print(selected_row)

        rate = target_num_words / selected_row["Words"]
        pipeline = [
            ParquetReader(
                f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
                read_metadata=False,
            ),
            SamplerFilter(rate=rate, seed=42),
            ParquetWriter(f"{output_path}/data/fineweb2_{language}"),
        ]

    elif language == "eng_Latn":
        rate = (
            0.01 * target_num_words / 12847061986
        )  # 12.8B words based on a subset of ablation :)
        pipeline = [
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb-edu",
                glob_pattern="data/*/*.parquet",
                read_metadata=False,
            ),
            SamplerFilter(rate=rate, seed=42),
            ParquetWriter(f"{output_path}/data/fineweb_edu"),
        ]
    elif language == "code":
        rate = (
            0.01 * target_num_words / 629431122
        )  # 628M words based on a subset of ablation :)
        pipeline = [
            ParquetReader(
                "hf://datasets/bigcode/starcoderdata",
                glob_pattern="**/*.parquet",
                text_key="content",
            ),
            LambdaFilter(
                lambda doc: doc.metadata["max_stars_count"] >= 2
                if "max_stars_count" in doc.metadata
                else True,
            ),
            remove_metadata,
            SamplerFilter(rate=rate, seed=42),
            ParquetWriter(f"{output_path}/data/starcoder"),
        ]
    else:
        raise NotImplementedError(f"Language {language} not implemented")

    print(f"\nSampler rate: {rate}")

    main_processing_executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        sbatch_args={"account": "qgz@cpu"},
        tasks=50,
        cpus_per_task=2,
        time="05:00:00",
        qos="qos_cpu-t3",
        partition="prepost",
        env_command="source ~/OpenLLM-BPI-Training/data/set_env.sh",
        logging_dir=f"{output_path}/logs/{language}",
        job_name=language,
    )

    main_processing_executor.run()
