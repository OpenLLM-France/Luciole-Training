from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.jeanzay import JZSlurmPipelineExecutor

if __name__ == "__main__":
    OUTPUT_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training/starcoder"

    main_processing_executor = JZSlurmPipelineExecutor(
        job_name=f"starcoder",
        pipeline=[ 
            HuggingFaceDatasetReader(
                "bigcode/starcoderdata", 
                limit=-1
                ),
            LambdaFilter(
                lambda doc: doc.metadata['max_stars_count'] >=2,
                exclusion_writer=JsonlWriter(
                    f"{OUTPUT_PATH}/1_low_stars_count" 
                    )
                ),
            JsonlWriter(f"{OUTPUT_PATH}/output")
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=50, 
        cpus_per_task=2,
        time="05:00:00",
        logging_dir=f"{OUTPUT_PATH}/logs",
        qos="qos_cpu-t3",
        partition="prepost",
        condaenv="datatrove-env",
    )

    main_processing_executor.run()
