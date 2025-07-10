from utils import create_parser, parse_args, create_executor, add_sampler_filter
import os
import re

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.data import DocumentsPipeline



def get_age(field, default=None):
    if not field:
        return default
    return int(field)

global _authors_info
_authors_info = {}

def filter_gutenberg(x, language, current_year=2025):
    global _authors_info
    death = get_age(x["authoryearofdeath"])
    birth = get_age(x["authoryearofbirth"])
    thr = 80 if language == "fr" else 70
    author = x["author"]
    if author not in _authors_info:
        _authors_info[author] = (birth, death)
    else:
        if _authors_info[author] != (birth, death):
            info = f"{_authors_info[author]} != {(birth, death)}"
            print(f"Author {author} has multiple birth/death dates: {info}")
            if not death and not birth:
                birth, death = _authors_info[author]
    if not death and not birth:
        # print(f"Unknown birth/dead date for {author}")
        age_ok = True
    else:
        age_ok = bool(
            (death and death <= current_year - thr)
            or (not death and birth and birth <= current_year - thr - 80)
        )
    copyright_ok = "copyright" != x["usagerights"]
    # if language == "en":
    #     return copyright_ok
    return age_ok and copyright_ok


def clean_text_pipeline(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        document.text = clean_text(document.text)
        yield document


def clean_text(text):
    def remove_gallica_mention(text):
        return re.sub("[^\n]*http://gallica.bnf.fr[^\n]*\n", "", text)

    def remove_licence(text):
        pattern = r"\n\n            \*\*\* END OF THE PROJECT GUTENBERG EBOOK.*"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    text = remove_gallica_mention(text)
    text = remove_licence(text)
    return text



if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "language",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/data-server/datasets/text/raw/multilang/Gutenberg/jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data-server/datasets/text/raw/multilang/Gutenberg/filtered_jsonl",
    )
    args = parse_args(parser)

    language = args.language
    input_folder = os.path.join(args.root_path, language)

    pipeline = [
        JsonlReader(input_folder),
        LambdaFilter(
            lambda doc: filter_gutenberg(doc.metadata, doc.metadata["language"]),
        ),
        clean_text_pipeline,
        JsonlWriter(
            os.path.join(args.output_path, "data", language),
            output_filename="data_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=os.path.join(args.output_path, "logs", language),
        job_name="gutenberg",
        tasks=len(os.listdir(input_folder)),
    )

    main_processing_executor.run()
