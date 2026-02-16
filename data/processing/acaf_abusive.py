import os

from utils import create_parser, parse_args, create_executor

_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from functools import partial


def simple_query_builder(runner, document):
    PROMPT = """Transformez le passage ci-dessous en un dialogue à plusieurs reprises entre un professeur de français et un étudiant. 
Le dialogue commence par la question de l'étudiant concernant d'éventuelles utilisations incorrectes de mots ou d'expressions françaises. 
Le professeur explique les utilisations incorrectes ou imprécises de ces termes en français, transmettant les détails du passage de manière conversationnelle, engageant parfois l'étudiant en lui posant des questions. L'étudiant est très intéressé et curieux, et demande parfois des éclaircissements ou plus de détails. 
À la fin du dialogue, l'étudiant a une compréhension plus approfondie et plus précise de la langue française.

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
        default=os.path.join(os.environ.get("HF_MODELS_CACHE", "/lustre/fsmisc/dataset/HuggingFace_Models"), "mistralai/Mixtral-8x22B-Instruct-v0.1"),
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
            # "/lustre/fsn1/projects/rech/qgz/uzq54wg/test.jsonl",
            "/lustre/fsn1/projects/rech/qgz/uhm96nw/mixtral_data_generation/edu_french",
            glob_pattern="acaf_sensabusives_prompts*.jsonl",
            text_key="question",
        ),
        partial(preproc, model_name=args.model_name),
        InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/acaf_abusive/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/acaf_abusive/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
        ),
    ]

    inference_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/acaf_abusive/logs",
        job_name="acaf_abusive",
        tasks=1,
        time="05:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        sbatch_args={
            "account": os.environ.get("SLURM_ACCOUNT_GPU", "wuh@h100"),
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
