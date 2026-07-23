from utils import (
    create_parser,
    parse_args,
    create_executor,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter
import random

# (comma_repeats, comma_cooldown_repeats)
repeats = {
    "arxiv_abstracts_filtered":(6, 0),
    "arxiv_papers_filtered":(6, 0.5),
    "biodiversity_heritage_library_filtered":(0.25, 0),
    "caselaw_access_project_filtered":(1, 0),
    "data_provenance_initiative_filtered":(6, 2), 
    "doab_filtered":(6, 2),
    "foodista_filtered":(6, 2),
    "github_archive_filtered":(6, 0),
    "library_of_congress_filtered":(0.25, 0),
    "libretexts_filtered":(6, 2),
    "news_filtered":(6, 2),
    "oercommons_filtered":(6, 2),
    "peS2o_filtered":(6, 0.1),
    "pre_1929_books_filtered":(1, 0),
    "pressbooks_filtered":(6, 2),
    "public_domain_review_filtered":(6, 2),
    "pubmed_filtered":(1, 0),
    "python_enhancement_proposals_filtered":(6, 2),
    "regulations_filtered":(6, 0),
    "stackexchange_filtered":(6, 0.25),
    "ubuntu_irc_filtered":(6, 0),
    "youtube_filtered":(1, 0),
}
# (comma_repeats, comma_cooldown_repeats)
# CC Common Crawl - (6, 0.3)
# Project Gutenberg - (1, 0)
# Stack V2 - (2, 0.1)
# UK Hansard - (6, 0)
# USGPO - (0.25, 0)
# USPTO - (0.25, 0)
# Wikimedia - (6, 0.4)
# Wikiteam - (4, 0)

def reweighting_function(doc):
    doc.metadata["source"] = source = doc.metadata["source"].split("/")[-1]
    main_ratio, cooldown_ratio = (r / 8 for r in repeats[source])
    if random.random() < main_ratio:
        doc.metadata["phase"] = "main"
        return True
    elif random.random() < main_ratio + cooldown_ratio:
        doc.metadata["phase"] = "cooldown"
        return True
    return False

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        ParquetReader(
            "hf://datasets/OpenLLM-France/Luciole-Training-Dataset/data/common_pile",
        ),
        LambdaFilter(
            filter_function=reweighting_function,
        ),
        JsonlWriter(
            f"{DATA_PATH}/common_pile_reweighted/data",
            output_filename="${phase}/${source}/rank${rank}.jsonl.gz",
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/common_pile_reweighted/logs",
        job_name="cp_reweight",
        tasks=20,
        skip_completed=not args.force,
    )
    main_processing_executor.run()
