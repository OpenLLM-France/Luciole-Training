from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from web_utils import get_web_pipeline, ROBOTSTXT_PATH
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

LANGUAGES = [
    "arb_Arab",
    "deu_Latn",
    "fra_Latn",
    "ita_Latn",
    "nld_Latn",
    "por_Latn",
    "spa_Latn",
]


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
        "--language",
        type=str,
        default="fra_Latn",
        help="Language to process",
        choices=LANGUAGES,
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    language = args.language

    if not args.push_only:
        pipeline = [
            ParquetReader(
                f"hf://datasets/epfml/FineWeb2-HQ/{language}",
            ),
            AssignCluster(),
            *get_web_pipeline(
                language,
                robots_txt_path=ROBOTSTXT_PATH,
                output_path=f"{DATA_PATH}/fineweb2_hq_filtered/{language}",
                do_edu=True,
                do_pii=True,
                do_decont=False,
            ),
            JsonlWriter(
                f"{DATA_PATH}/fineweb2_hq_filtered/{language}/data",
                output_filename="${cluster_size_group}_edu_${edu_score}_rank${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/fineweb2_hq_filtered/{language}/logs",
            job_name=f"fw_hq_{language}",
            partition="prepost",
            cpus_per_task=1,
            time="20:00:00",
        )
        main_executor.run()

    else:

        def fix_data(
            data: DocumentsPipeline,
            rank: int = 0,
            world_size: int = 1,
            language: str = None,
        ) -> DocumentsPipeline:
            from web_utils import LanguageCodes

            for doc in data:
                doc.metadata["language_iso"] = LanguageCodes.fineweb_to_iso1(language)
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/fineweb2_hq_filtered/{language}/data",
            ),
            partial(fix_data, language=language),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/fineweb2_hq_filtered/{language}/data_hf",
                output_filename="data/fineweb2_hq/${language_iso}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="fineweb2_hq",
                    id_key=None,
                    language=None,
                    language_key="language_iso",
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
            logging_dir=f"{DATA_PATH}/fineweb2_hq_filtered/{language}/logs_hf",
            job_name="hf_fw2_hq",
            tasks=5,
            skip_completed=not args.force,
        )

        hf_executor.run()
