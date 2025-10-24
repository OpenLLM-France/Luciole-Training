from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter


class MergeDocument(PipelineStep):
    type = "📑  MERGE"
    name = "📑  Merge Document"

    def __init__(
        self,
        src_language,
        tgt_language,
        revert_prob=0.5,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.revert_prob = revert_prob
        self.exclusion_writer = exclusion_writer

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import contextlib
        from datatrove.data import Document
        import random

        def breaks_document(chunk, frequency_cutoff, lid_cutoff):
            line = chunk.metadata
            if "None" in [
                line["src_paragraph_id"],
                line["src_sentence_id"],
                line["src_start_id"],
                line["src_end_id"],
                line["tgt_paragraph_id"],
                line["tgt_sentence_id"],
                line["tgt_start_id"],
                line["tgt_end_id"],
            ]:
                return True
            if int(line["duplication_count"]) > frequency_cutoff:
                return True
            if float(line["src_lid_prob"]) < lid_cutoff:
                return True
            if float(line["tgt_lid_prob"]) < lid_cutoff:
                return True
            if len(line["src"].strip()) == 0:
                return True
            if len(line["tgt"].strip()) == 0:
                return True
            return False

        def translate(text, prompt_language):
            translations = {
                "en": {
                    "en": "English",
                    "fr": "French",
                    "es": "Spanish",
                    "de": "German",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "nl": "Dutch",
                },
                "fr": {
                    "en": "anglais",
                    "fr": "français",
                    "es": "espagnol",
                    "de": "allemand",
                    "it": "italien",
                    "pt": "portuguais",
                    "nl": "néerlandais",
                },
            }
            try:
                return translations.get(prompt_language, "fr")[text]
            except KeyError:
                raise NotImplementedError(
                    f"No translation for '{text}' in '{prompt_language}'"
                )

        def get_prompt(src_language, prompt_language):
            # Build the file path
            prompt_language = "fr" if random.random() < 0.5 else "en"
            file_path = f"assets/paradocs_prompt_{prompt_language}.txt"
            # Read all lines (stripping trailing newlines/spaces)
            with open(file_path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            # Draw one prompt uniformly at random
            prompt = random.choice(prompts).format(
                language=translate(src_language, prompt_language)
            )
            return prompt

        doc = None
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for chunk in data:
                chunk.metadata["src"] = chunk.text
                # Init new doc
                if doc is None:
                    do_revert = random.random() < self.revert_prob
                    if do_revert:
                        prompt = get_prompt(src_language, tgt_language)
                    else:
                        prompt = get_prompt(tgt_language, src_language)

                    doc = Document(id=chunk.id, text=[], metadata=chunk.metadata)
                    doc.text.append({"role": "system", "content": prompt})

                # Continue
                if chunk.metadata["src_docid"] == doc.metadata["src_docid"]:
                    if breaks_document(chunk, frequency_cutoff=100, lid_cutoff=0.5):
                        if self.exclusion_writer:
                            writer.write(chunk, rank)
                    elif not do_revert:
                        doc.text.extend(
                            [
                                {"role": "user", "content": chunk.text},
                                {"role": "assistant", "content": chunk.metadata["tgt"]},
                            ]
                        )
                    else:
                        doc.text.extend(
                            [
                                {"role": "user", "content": chunk.metadata["tgt"]},
                                {"role": "assistant", "content": chunk.text},
                            ]
                        )

                # break
                else:
                    yield doc
                    doc = None


def apply_chat_template(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from transformers import AutoTokenizer

    tokenizer_name = "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for doc in data:
        doc.metadata["conversation"] = doc.text
        doc.text = tokenizer.apply_chat_template(
            doc.text, tokenize=False, enable_thinking=False
        )
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--languages", default="en-fr", type=str)
    parser.add_argument("--revert", default=0.5, type=float)
    args = parse_args(parser)
    src_language, tgt_language = args.languages.split("-")
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "jhu-clsp/paradocs",
            {
                "name": f"{args.languages}-strict",
                "split": "train",
                "trust_remote_code": True,
            },
            streaming=True,
            text_key="src",
        ),
        MergeDocument(
            src_language=src_language,
            tgt_language=tgt_language,
            revert_prob=args.revert,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/paradocs/{args.languages}_revert{args.revert}/removed",
            ),
        ),
        apply_chat_template,
        JsonlWriter(
            f"{DATA_PATH}/paradocs/{args.languages}_revert{args.revert}/data",
            output_filename="${collection}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/paradocs/{args.languages}_revert{args.revert}/logs",
        job_name="paradocs",
        tasks=20,
        partition="prepost",
        time="20:00:00",
        cpu_per_task=4,
    )

    filter_executor.run()

# python paradocs.py --languages en-fr --revert 0
# python paradocs.py --languages en-fr --revert 1
# python paradocs.py --languages en-fr --revert 0.5
# python paradocs.py --languages en-es --revert 0
# python paradocs.py --languages en-es --revert 1
# python paradocs.py --languages en-es --revert 0.5
# python paradocs.py --languages en-nl --revert 0
# python paradocs.py --languages en-nl --revert 1
# python paradocs.py --languages en-nl --revert 0.5
# python paradocs.py --languages en-de --revert 0
# python paradocs.py --languages en-de --revert 1
# python paradocs.py --languages en-de --revert 0.5
# python paradocs.py --languages en-pt --revert 0
# python paradocs.py --languages en-pt --revert 1
# python paradocs.py --languages en-pt --revert 0.5
# python paradocs.py --languages en-it --revert 0
# python paradocs.py --languages en-it --revert 1
# python paradocs.py --languages en-it --revert 0.5
