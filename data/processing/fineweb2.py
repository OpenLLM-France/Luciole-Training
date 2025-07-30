from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import get_web_pipeline, ROBOTSTXT_PATH


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

    pipeline = [
        ParquetReader(
            f"/lustre/fsmisc/dataset/HuggingFace/HuggingFaceFW/fineweb-2/data/{language}/train"
            if args.jz
            else f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
        ),
        AssignCluster(),
        *get_web_pipeline(
            language.removesuffix("_removed"),
            robots_txt_path=ROBOTSTXT_PATH,
            output_path=f"{DATA_PATH}/fineweb2_filtered/{language}",
            do_edu=True,
            do_pii=True,
            do_decont=False,
        ),
        PrefixFormatter(
            date_keys=["date"],
            date_format="%Y-%m-%dT%H:%M:%SZ",
            prefix_pipeline={"domain": "Domain", "date": "Date"},
        ),
        JsonlWriter(
            f"{DATA_PATH}/fineweb2_filtered/{language}/data",
            output_filename="${cluster_size_group}_edu_${edu_score}_rank${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    if language in [
        "arb_Arab",
        "deu_Latn",
        "fra_Latn",
        "fra_Latn_removed",
        "ita_Latn",
        "nld_Latn",
        "por_Latn",
        "cat_Latn",
    ]:
        tasks = 50
    else:
        tasks = 1

    main_executor = create_executor(
        pipeline,
        tasks=tasks,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb2_filtered/{language}/logs",
        job_name=f"fw_{language}",
        partition="prepost",
        cpus_per_task=1,
        time="20:00:00",
    )
    main_executor.run()
