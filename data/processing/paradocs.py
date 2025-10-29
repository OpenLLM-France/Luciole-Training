from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter


def merge_document(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from datatrove.data import Document

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

    doc = None
    for chunk in data:
        chunk.metadata["src"] = chunk.text
        # Init new doc
        if doc is None:
            doc = Document(id=chunk.id, text=[], metadata={})
            doc.metadata["src_docid"] = chunk.metadata["src_docid"]
            doc.metadata["collection"] = chunk.metadata["collection"]
            doc.metadata["tgt"] = []

        # Continue
        if chunk.metadata["src_docid"] == doc.metadata["src_docid"]:
            if breaks_document(chunk, frequency_cutoff=100, lid_cutoff=0.5):
                pass
            else:
                doc.text.append(chunk.text)
                doc.metadata["tgt"].append(chunk.metadata["tgt"])

        # break
        else:
            yield doc
            doc = None


class ParadocsProcessing(PipelineStep):
    def __init__(
        self,
        src_language,
        tgt_language,
        revert_prob=0.5,
        geometric_param=1.0 / 3,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.revert_prob = revert_prob
        self.geometric_param = geometric_param
        self.exclusion_writer = exclusion_writer

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import random

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

        def get_prompt(src_language):
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

        def chunk_parallel_lists(src, tgt):
            import numpy as np

            assert len(src) == len(tgt), "src and tgt must have the same length"

            src_chunks, tgt_chunks = [], []
            i = 0
            n = len(src)

            while i < n:
                # choose random chunk length between min_size and max_size
                chunk_size = np.random.geometric(self.geometric_param, size=None)
                src_chunks.append(src[i : i + chunk_size])
                tgt_chunks.append(tgt[i : i + chunk_size])
                i += chunk_size

            src_chunks = ["\n".join(s) for s in src_chunks]
            tgt_chunks = ["\n".join(s) for s in tgt_chunks]
            return src_chunks, tgt_chunks

        for doc in data:
            prompt = get_prompt(self.src_language)
            do_revert = random.random() < self.revert_prob
            if not do_revert:
                src = doc.text
                tgt = doc.metadata.pop("tgt")
            else:
                src = doc.metadata.pop("tgt")
                tgt = doc.text

            # Merge chunk
            src, tgt = chunk_parallel_lists(src, tgt)

            # Conversation template
            src[0] = prompt + "\n\n" + src[0]
            # doc.text = [{"role": "system", "content": prompt}]
            # num_of_chunk_in_current_turn = np.random.poisson(3)
            for user_turn, assistant_turn in zip(src, tgt):
                doc.text.extend(
                    [
                        {"role": "user", "content": user_turn},
                        {"role": "assistant", "content": assistant_turn},
                    ]
                )
            yield doc


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
    parser.add_argument("--geometric_param", default=1.0 / 3, type=float)
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
        merge_document,
        ParadocsProcessing(
            src_language=src_language,
            tgt_language=tgt_language,
            revert_prob=args.revert,
        ),
        apply_chat_template,
        JsonlWriter(
            f"{DATA_PATH}/paradocs_geom{args.geometric_param:.2f}/{args.languages}_revert{args.revert}/data",
            output_filename="${collection}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/paradocs_geom{args.geometric_param:.2f}/{args.languages}_revert{args.revert}/logs",
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
