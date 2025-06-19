import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import PerplexityFilter

TASKS = 1
SUPPORTED_YEARS = [
    "1770",
    "1771",
    "1772",
    "1773",
    "1774",
    "1777",
    "1778",
    "1779",
    "1791",
    "1792",
    "1793",
] + [str(year) for year in range(1796, 1964 + 1)]

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "american_stories"

    for year in SUPPORTED_YEARS:
        year = str(year)

        output_path = os.path.join(DATA_PATH, dataset_name)
        pipeline = [
            HuggingFaceDatasetReader(
                "dell-research-harvard/AmericanStories",
                {
                    "name": "subset_years",
                    "trust_remote_code": True,
                    "year_list": [year],
                    "split": year,
                },
                text_key="article",
                streaming=True,
            ),
            PerplexityFilter(
                language="en",
                min_ppl=10,
                max_ppl=2000,
                exclusion_writer=JsonlWriter(f"{output_path}/removed/ppl/{year}"),
            ),
            JsonlWriter(f"{output_path}/data/{year}"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            logging_dir=f"{output_path}/logs/{year}",
            job_name=dataset_name,
            tasks=TASKS,
        )
        main_processing_executor.run()
