from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.jeanzay import JZSlurmPipelineExecutor
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--debug", action="store_true", help="Debug mode")

if __name__ == "__main__":
    args = parser.parse_args()

    OUTPUT_PATH = "/lustre/fsn1/projects/rech/qgz/commun/datasets/training/algebraic_stack"
    if args.debug:
        OUTPUT_PATH += '/debug'

    main_processing_executor = JZSlurmPipelineExecutor(
        job_name=f"algebraic_stack",
        pipeline=[ 
            HuggingFaceDatasetReader(
                "EleutherAI/proof-pile-2", 
                {"name": "algebraic-stack", "trust_remote_code":True},
                limit=1000 if args.debug else -1
                ),
            JsonlWriter(f"{OUTPUT_PATH}/output")
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
