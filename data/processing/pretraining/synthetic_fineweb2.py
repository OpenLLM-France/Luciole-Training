import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_web_pipeline, ROBOTSTXT_PATH
from datatrove.pipeline.filters import (
    GopherRepetitionFilter,
)
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def get_prompt_level(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        # Open type
        if (
            "very small vocabulary and extremely simple sentences"
            in doc.metadata["instruction"]
        ):
            prompt_level = "easy"
        elif (
            "using high quality French language such as in sentences on Wikipedia"
            in doc.metadata["instruction"]
        ):
            prompt_level = "medium"
        elif (
            "using very terse and abstruse language that only an erudite scholar will understand"
            in doc.metadata["instruction"]
        ):
            prompt_level = "hard"
        else:
            prompt_level = "unknown"
        doc.metadata["prompt_level"] = prompt_level
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            JsonlReader(
                os.path.join(
                    os.environ.get("OpenLLM_OUTPUT", ""),
                    "data/raw_data/full_datasets/synthetic_fineweb2_fr_extract_knowledge_with_urls",
                ),
            ),
            *get_web_pipeline(
                "fr",
                robots_txt_path=ROBOTSTXT_PATH,
                output_path=f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered",
                do_edu=True,
                do_pii=False,
                do_decont=False,
            ),
            get_prompt_level,
            GopherRepetitionFilter(
                language="fr",  # [!] THIS IS IMPORTANT: we need this to know which word tokenizer to use to split
                # into words and ngrams
                # we disable these. trafilatura pretty much removes paragraph and we use a different threshold
                # for dup_line_char_frac in fineweb quality
                dup_para_frac=0,
                dup_line_char_frac=0,
                dup_para_char_frac=0,
                dup_line_frac=0.264,
                top_n_grams=((2, 0.161), (3, 0.149), (4, 0.134)),
                dup_n_grams=(
                    (5, 0.15),
                    (6, 0.141),
                    (7, 0.131),
                    (8, 0.121),
                    (9, 0.111),
                    (10, 0.1),
                ),
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/removed/goph_rep/"
                ),
            ),
            JsonlWriter(
                f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/data",
                output_filename="${prompt_level}_edu_${edu_score}_rank${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/logs",
            job_name="synthetic",
            tasks=50,
        )
        main_processing_executor.run()

    else:
        for prompt_level in ["medium", "hard"]:
            pipeline = [
                JsonlReader(
                    f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/data",
                    glob_pattern=prompt_level + "_*.jsonl.gz",
                ),
                HuggingFaceDatasetWriter(
                    dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                    + ("-debug" if args.debug else ""),
                    private=True,
                    local_working_dir=f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/data_hf/{prompt_level}",
                    output_filename="data/synth_fineweb2/fr/" + "${rank}.parquet",
                    adapter=partial(
                        _custom_adapter_for_hf,
                        source="synth_fineweb2",
                        id_key=None,
                        reset_id=False,
                        language="fr",
                        language_key=None,
                        conversation_key=None,
                        remove_keys=[],
                    ),
                    cleanup=True,
                    expand_metadata=False,
                    schema=HF_SCHEMA,
                ),
            ]

            hf_executor = create_executor(
                pipeline,
                local=args.local,
                debug=args.debug,
                logging_dir=f"{DATA_PATH}/synthetic_fineweb2_fr_extract_knowledge_filtered/logs_hf/{prompt_level}",
                job_name="hf_synth_fw",
                tasks=1,
            )

            hf_executor.run()
