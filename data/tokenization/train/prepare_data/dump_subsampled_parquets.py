from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter, LambdaFilter
from datatrove.pipeline.writers import ParquetWriter
from datatrove.data import DocumentsPipeline
import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.abspath("../../../processing"))
from utils import get_edu_pipeline  # noqa: E402


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


def main(
    language: str,
    target_num_words: int,
    output_path: str = None,
    fasttext_path: str = None,
    slurm: bool = False,
    tasks: int = 50,
    limit=-1,
):
    # Read stats from fineweb2

    if language == "eng_Latn":
        rate = (
            0.01 * target_num_words / 12_847_061_986
        )  # 12.8B words based on a subset of ablation :)
        pipeline = [
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb-edu",
                glob_pattern="data/*/*.parquet",
                read_metadata=False,
                limit=limit,
            ),
            SamplerFilter(rate=rate, seed=42),
            remove_metadata,
            ParquetWriter(f"{output_path}/fineweb_edu"),
        ]
    elif language == "code":
        rate = (
            0.01 * target_num_words / 629_431_122
        )  # 628M words based on a subset of ablation :)
        pipeline = [
            ParquetReader(
                "hf://datasets/bigcode/starcoderdata",
                glob_pattern="**/*.parquet",
                text_key="content",
                limit=limit,
            ),
            LambdaFilter(
                lambda doc: doc.metadata["max_stars_count"] >= 2
                if "max_stars_count" in doc.metadata
                else True,
            ),
            SamplerFilter(rate=rate, seed=42),
            remove_metadata,
            ParquetWriter(f"{output_path}/starcoder"),
        ]

    else:
        df = read_markdown_table("fineweb2_languages.md")
        selected_row = df[df["Subset"] == language]
        assert (
            len(selected_row) > 0
        ), f"Language {language} not found in the table (is it in FineWeb2?)."
        assert len(selected_row) == 1
        selected_row = selected_row.iloc[0]

        if fasttext_path is not None:
            try:
                exclusion_writer = ParquetWriter(
                    f"{output_path}/removed/fineweb2_{language}"
                )
                edu_pipeline = get_edu_pipeline(fasttext_path, exclusion_writer)
            except (FileNotFoundError, OSError):
                edu_pipeline = []

        rate = target_num_words / selected_row["Words"]
        pipeline = [
            ParquetReader(
                f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
                read_metadata=False,
                limit=limit,
            ),
            SamplerFilter(rate=rate, seed=42),
            *edu_pipeline,
            remove_metadata,
            ParquetWriter(f"{output_path}/fineweb2_{language}"),
        ]

    print(f"\nSampler rate: {rate}")
    assert (
        rate <= 1
    ), f"Target number of words ({target_num_words}) is too high for language {language} with {selected_row['Words']} words in the dataset."

    if slurm:
        main_processing_executor = SlurmPipelineExecutor(
            pipeline=pipeline,
            sbatch_args={"account": "qgz@cpu"},
            tasks=tasks,
            cpus_per_task=2,
            time="05:00:00",
            qos="qos_cpu-t3",
            partition="prepost",
            env_command="source ../../../../data/set_env.sh",
            logging_dir=f"{output_path}/logs",
            job_name=language,
        )
    else:
        main_processing_executor = LocalPipelineExecutor(pipeline=pipeline, tasks=tasks)

    main_processing_executor.run()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--language",
        type=str,
        default="fra_Latn",
        help="Language to process",
    )
    argparser.add_argument(
        "--target",
        type=int,
        default=1e9,
        help="Target number of words in the dataset",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            os.getenv("OpenLLM_OUTPUT", os.getenv("HOME")), "data/data_for_tokenization"
        ),
        help="Output parent path",
    )
    argparser.add_argument(
        "--fasttext_path",
        type=str,
        default=None,
        help="Path to the fasttext path to apply if any (must be the edu scorer)",
    )
    argparser.add_argument(
        "--slurm",
        action="store_true",
        help="Run the pipeline using Slurm",
    )
    argparser.add_argument(
        "--tasks",
        type=int,
        default=48,
        help="Number of tasks to use in Slurm",
    )
    argparser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of samples to read (debug)",
    )

    args = argparser.parse_args()

    main(
        args.language,
        args.target,
        args.output_path,
        fasttext_path=args.fasttext_path,
        slurm=args.slurm,
        tasks=args.tasks,
        limit=args.limit,
    )
