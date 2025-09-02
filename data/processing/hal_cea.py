from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--path",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/hal_cea",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        JsonlReader(args.path),
        LambdaFilter(
            lambda doc: doc.metadata["nb_formula_errors"] / max(1, doc.metadata["nb_formulas"]) < 0.2,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/hal_cea_filtered/removed"
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/hal_cea_filtered/data",
            output_filename="${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/hal_cea_filtered/logs",
        job_name="hal_cea_filtered",
        tasks=5,
    )

    main_processing_executor.run()
