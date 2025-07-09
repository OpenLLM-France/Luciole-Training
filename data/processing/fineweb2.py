from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import (
    get_edu_filters,
    get_pii_formatter,
    get_decontamination_filters,
)


class AssignCluster(PipelineStep):
    type = "✂️ - FORMAT"
    name = "👬🏻 Assign Duplication Cluster"

    @staticmethod
    def get_cluster_size_group(cluster_size):
        if cluster_size in [1, 2, 3, 4]:
            return f"cluster_size-{cluster_size}"
        elif cluster_size < 100:
            return "cluster_size-5-100"
        elif cluster_size < 1000:
            return "cluster_size-100-1000"
        else:
            return "cluster_size-1000+"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        for doc in data:
            cluster_size_group = self.get_cluster_size_group(
                doc.metadata["minhash_cluster_size"]
            )
            doc.metadata["cluster_size_group"] = cluster_size_group
            yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    parser.add_argument(
        "--jz",
        action="store_true",
        help="Use jz version of the fineweb2 dataset",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    language = args.language

    ############
    # Filter Fineweb2 DATASET
    ############

    # Get language specific filtering and formatting
    edu_filters = get_edu_filters(language)
    pii_formatter = get_pii_formatter(language)

    pipeline = [
        ParquetReader(
            f"/lustre/fsmisc/dataset/HuggingFace/HuggingFaceFW/fineweb-2/data/{language}/train"
            if args.jz
            else f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
        ),
        AssignCluster(),
        *edu_filters,
        *pii_formatter,
        PrefixFormatter(date_keys=["date"], date_format="%Y-%m-%dT%H:%M:%SZ"),
        JsonlWriter(
            f"{DATA_PATH}/fineweb2_filtered/{language}/data",
            output_filename="${cluster_size_group}_edu_${edu_score}_rank${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_executor = create_executor(
        pipeline,
        tasks=50,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb2_filtered/{language}/logs",
        job_name="fineweb2_filtered",
        partition="cpu_p1" if args.jz else "prepost",
        cpus_per_task=2,  # OOM with 1...
        time="20:00:00",
    )
    main_executor.run()

    ############
    # Decontaminate Fineweb2 DATASET
    ############

    decontamination_filters = get_decontamination_filters(language)

    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/fineweb2_filtered/{language}/data",
        ),
        *decontamination_filters,
        PrefixFormatter(date_keys=["timestamp"], date_format="%Y/%m/%d %H:%M:%S"),
        JsonlWriter(
            f"{DATA_PATH}/fineweb2_decont/{language}/data",
            output_filename="${source}_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    decont_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb2_decont/{language}/logs",
        job_name="fineweb2_decont",
        tasks=50,
        partition="cpu_p1",
        time="20:00:00",
        depends=main_executor,
    )
