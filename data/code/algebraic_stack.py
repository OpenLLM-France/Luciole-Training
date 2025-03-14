from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.jeanzay import JZSlurmPipelineExecutor

if __name__ == "__main__":
    OUTPUT_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training/algebraic_stack"

    main_processing_executor = JZSlurmPipelineExecutor(
        job_name=f"algebraic_stack",
        pipeline=[ 
            HuggingFaceDatasetReader(
                "EleutherAI/proof-pile-2", 
                {'name': "algebraic-stack"},
                limit=-1
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
