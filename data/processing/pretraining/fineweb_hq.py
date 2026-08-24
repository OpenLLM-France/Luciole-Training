from utils import create_parser, parse_args, create_executor
from web_utils import get_web_pipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter
from numpy.random import default_rng
from functools import partial

_rng = default_rng(42)


def keep_rate(rounded_score):
    """Linear subsampling rate: 0.50 -> 0, 0.55 -> 0.1, ..., 0.95 -> 0.9, 1.00 -> 1."""
    return min(max((rounded_score - 0.5) / 0.5, 0.0), 1.0)


def process_score(doc, subsample=False):
    score = doc.metadata.get("quality_score", 0.0)
    rounded = round(score / 0.05) * 0.05
    doc.metadata["quality_score_rounded"] = f"{rounded:.2f}"
    if subsample:
        return _rng.uniform() < keep_rate(rounded)
    return score > 0.5


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--subsample",
        action="store_true",
        help="Subsample documents with a keep rate linear in the rounded quality "
        "score (0.50 -> 0, 0.55 -> 0.1, ..., 1.00 -> 1) instead of keeping "
        "everything above 0.5",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    ### LOAD
    pipeline = [
        ParquetReader(
            "hf://datasets/epfml/FineWeb-HQ/data",
        ),
        LambdaFilter(partial(process_score, subsample=args.subsample)),
        *get_web_pipeline(
            "en",
            f"{DATA_PATH}/fineweb_hq_filtered",
            do_edu=False,
            do_pii=True,
            do_decont=False,
        ),
        JsonlWriter(
            f"{DATA_PATH}/fineweb_hq_filtered/data",
            output_filename="quality_${quality_score_rounded}_rank${rank}.jsonl.gz",
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb_hq_filtered/logs",
        job_name="fw-hq",
        tasks=200,
        time="20:00:00",
    )
    main_processing_executor.run()
