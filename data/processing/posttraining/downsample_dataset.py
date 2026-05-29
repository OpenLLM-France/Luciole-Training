from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import SamplerFilter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "input_path",
        type=str
    )
    parser.add_argument(
        "output_path",
        type=str
    )
    parser.add_argument(
        "--global_pattern",
        type=str
    )
    parser.add_argument(
        "--rate",
        default=0.1,
        type=float
    )
    args = parse_args(parser)

    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.global_pattern
        ),
        SamplerFilter(rate=args.rate, seed=42),
        JsonlWriter(
            f"{args.output_path}_rate{args.rate}/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_path}_rate{args.rate}/logs",
        job_name=f"downsample",
        tasks=10,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )

    main_processing_executor.run()

# python downsample_dataset.py  
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/pleais_rag/openrag_thinking 
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/pleais_rag/openrag_thinking_downsample 
# --glob "data/en/*.jsonl.gz" --rate 0.125

# python downsample_dataset.py \
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/nemotron_agentic_sft_v2 \
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/nemotron_agentic_sft_v2_downsample \
# --glob "*/data/*.jsonl.gz" --rate 0.1 