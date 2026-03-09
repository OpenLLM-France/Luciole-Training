import os

from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from functools import partial


def simple_query_builder(runner, document):
    PROMPT = """Vous êtes un expert en langue française et expliquez les subtilités de l'orthographe française. Pour la question suivante sur l'orthographe correcte,  reformulez le passage donné afin de donner une réponse claire et détaillée.

Text:
{text}"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.format(text=document.text)},
                ],
            }
        ],
        "max_tokens": 2048,
    }


def preproc(data, rank, world_size, model_name):
    for doc in data:
        doc.text = doc.text.strip() + " : " + doc.metadata["answer"]
        doc.metadata["model_name"] = model_name
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/lustre/fsmisc/dataset/HuggingFace_Models/mistralai/Mixtral-8x22B-Instruct-v0.1",
    )
    parser.add_argument("--tp", type=int, default=1)
    args = parse_args(parser)
    DATA_PATH = args.data_path

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        # temperature=0.6,
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    pipeline = [
        JsonlReader(
            os.path.join(
                os.environ.get("DATA", ""), "mixtral_data_generation/edu_french"
            ),
            glob_pattern="lf_orthographe_prompts*.jsonl",
            text_key="question",
        ),
        partial(preproc, model_name=args.model_name),
        InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/lf_orthographe/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/lf_orthographe/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
        ),
    ]

    inference_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/lf_orthographe/logs",
        job_name="acaf_abusive",
        tasks=1,
        time="02:00:00",
        qos="qos_gpu_h100-dev",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command="source ~/OpenLLM-BPI-Training/data/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": f"gpu:{args.tp}",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
    )
    inference_executor.run()

# Example commands for debug in interactive mode:
# python acaf_abusive.py --model_name Qwen/Qwen3-0.6B --debug --local --force
# python acaf_abusive.py --model_name /lustre/fsmisc/dataset/HuggingFace_Models/mistralai/Mixtral-8x22B-Instruct-v0.1 --tp 4 --debug --local --force
# Run on full dataset with slurm on JZ (increase tasks if needed but keep it fix after):
# python acaf_abusive.py --model_name /lustre/fsmisc/dataset/HuggingFace_Models/mistralai/Mixtral-8x22B-Instruct-v0.1 --tp 4
