from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_web_pipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter


def process_score(doc):
    score = doc.metadata.get("quality_score", 0.0)
    rounded = round(score / 0.05) * 0.05
    doc.metadata["quality_score_rounded"] = f"{rounded:.2f}"
    return score > 0.5


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    ### LOAD
    pipeline = [
        ParquetReader(
            "hf://datasets/epfml/FineWeb-HQ/data",
        ),
        LambdaFilter(process_score),
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
    add_sampler_filter(pipeline, args.sample_rate)

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
