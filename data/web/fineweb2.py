from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.jeanzay import JZSlurmPipelineExecutor
import argparse

parser = argparse.ArgumentParser("Load Fineweb 2 dataset")
parser.add_argument(
    "--language", type=str, default="fra_Latn", help=""
)

if __name__ == "__main__":
    args = parser.parse_args()
    OUTPUT_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training"
    language = args.language

    main_processing_executor = JZSlurmPipelineExecutor(
        job_name=f"fineweb2",
        pipeline=[ 
            ParquetReader(
                f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train", 
                limit=-1
                ),
            JsonlWriter(f"{OUTPUT_PATH}/fineweb-2/data/{language}/train")
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=50, 
        cpus_per_task=10,
        time="02:00:00",
        logging_dir=f"{OUTPUT_PATH}/fineweb-2/logs/{language}/train",
        qos="qos_cpu-t3",
        partition="prepost",
        condaenv="datatrove-env",
    )

    main_processing_executor.run()
