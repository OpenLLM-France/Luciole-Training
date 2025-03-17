from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.jeanzay import JZSlurmPipelineExecutor
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--debug", action="store_true", help="Debug mode")

if __name__ == "__main__":
    args = parser.parse_args()
    
    OUTPUT_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training/starcoder"
    if args.debug:
        OUTPUT_PATH += '/debug'

    main_processing_executor = JZSlurmPipelineExecutor(
        job_name=f"starcoder",
        pipeline=[ 
            ParquetReader(
                f"hf://datasets/bigcode/starcoderdata", 
                glob_pattern = "**/*.parquet",
                text_key='content',
                limit=1000 if args.debug else -1
                ),
            LambdaFilter(
                lambda doc: doc.metadata['max_stars_count'] >=2,
                exclusion_writer=JsonlWriter(
                    f"{OUTPUT_PATH}/1_low_stars_count" 
                    )
                ),
            JsonlWriter(f"{OUTPUT_PATH}/1_high_stars_count")
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=1 if args.debug else 50, 
        cpus_per_task=2,
        time="05:00:00",
        logging_dir=f"{OUTPUT_PATH}/logs",
        qos="qos_cpu-t3",
        partition="prepost",
        condaenv="datatrove-env",
    )

    main_processing_executor.run()
