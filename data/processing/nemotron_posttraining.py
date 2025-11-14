import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
import re
import random
from functools import partial


def split_thinking(output):
    think_pattern = r"<think>(.*?)</think>"
    thoughts = re.findall(think_pattern, output, flags=re.DOTALL)
    # assert len(thoughts) in [0, 1]
    if len(thoughts) == 0:
        return "", output
    thoughts = thoughts[0]
    remaining_text = re.sub(think_pattern, "", output, flags=re.DOTALL).strip()

    return thoughts, remaining_text


def convert_messages(
    data: DocumentsPipeline,
    rank: int = 0,
    world_size: int = 1,
    keep_thinking=False,
    language="en",
    message_length=3,
) -> DocumentsPipeline:
    for document in data:
        message = document.metadata["messages"]
        assert (
            len(message) == message_length
        ), f"Expected {message_length} messages, got {len(message)}"
        user_content = message[-2]["content"]
        assistant_content = message[-1]["content"]
        thoughts, answer = split_thinking(assistant_content)

        if not keep_thinking:
            thoughts = ""

        problem_name = random.choice(
            {
                "en": ["Problem:", "Question:", "Query:", "Prompt:", "user:", ""],
                "fr": [
                    "Problème :",
                    "Question :",
                    "Requête :",
                    "Demande :",
                    "user:",
                    "",
                ],
                "es": [
                    "Problema:",
                    "Pregunta:",
                    "Consulta:",
                    "Solicitud:",
                    "user:",
                    "",
                ],
                "it": ["Problema:", "Domanda:", "Richiesta:", "Prompt:", "user:", ""],
                "de": ["Problem:", "Frage:", "Anfrage:", "Aufforderung:", "user:", ""],
            }[language]
        )
        # Thinking is always in English
        thoughts_name = (
            random.choice(["Thoughts:", "Reasoning:", "Thinking:"])
            if (problem_name not in ["", "user:"])
            else {"user:": "think:", "": ""}[problem_name]
        )
        solution_name = (
            random.choice(
                {
                    "en": ["Answer:", "Solution:", "Final Answer:"],
                    "fr": ["Réponse :", "Solution :", "Réponse finale :"],
                    "es": ["Respuesta:", "Solución:", "Respuesta final:"],
                    "it": ["Risposta:", "Soluzione:", "Risposta finale:"],
                    "de": ["Antwort:", "Lösung:", "Endgültige Antwort:"],
                }[language]
            )
            if (problem_name not in ["", "user:"])
            else {"user:": "assistant:", "": ""}[problem_name]
        )

        if not thoughts:
            document.text = (
                problem_name
                + "\n"
                + user_content
                + "\n"
                + solution_name
                + "\n"
                + answer
            ).strip()
        else:
            document.text = (
                problem_name
                + "\n"
                + user_content
                + "\n"
                + thoughts_name
                + "\n"
                + thoughts
                + "\n"
                + solution_name
                + "\n"
                + answer
            ).strip()

        yield document


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--subset",
        type=str,
        default="math",
        help="Subset to process",
        choices=[
            "stem",
            "chat",
            "code",
            "math",
            "multilingual_fr",
            "multilingual_es",
            "multilingual_it",
            "multilingual_de",
        ],
    )
    parser.add_argument(
        "--keep_thinking",
        action="store_true",
        help="Keep the thinking in the output",
        default=False,
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "nemotron_posttraining"
    subset_name = args.subset + ("_with_thinking" if args.keep_thinking else "")
    output_path = os.path.join(DATA_PATH, dataset_name, subset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/nvidia/Nemotron-Post-Training-Dataset-v2/data",
            glob_pattern=args.subset + "*.parquet",
            text_key="category",
            id_key="uuid",
        ),
        partial(
            convert_messages,
            keep_thinking=args.keep_thinking,
            language="en"
            if not args.subset.startswith("multilingual")
            else args.subset.split("_")[-1],
        ),
        JsonlWriter(f"{output_path}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
        tasks=50,
    )

    main_processing_executor.run()
