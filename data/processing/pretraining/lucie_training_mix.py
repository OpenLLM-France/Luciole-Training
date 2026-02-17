from utils import (
    create_parser,
    parse_args,
    create_executor,
)
from numpy.random import default_rng

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.writers import JsonlWriter

from datatrove.pipeline.readers import HuggingFaceDatasetReader


class SamplerFilter(BaseFilter):
    """
    Sample filter to randomly keep `rate`*100 percent of samples

    """

    name = "🎲 Sampler"

    def __init__(
        self,
        rate: float | None = 0.065,
        seed: int = None,
        exclusion_writer: DiskWriter = None,  # rate to KEEP
    ):
        """ """
        super().__init__(exclusion_writer)
        assert rate < 1 / 3
        self.rate = rate
        self.uniform = default_rng(seed).uniform

    def get_repeat(self, source, language):
        if source == "Eurovoc":
            if language == "en":
                return 1
            else:
                return 2
        if source == "Gutenberg":
            if language == "fr":
                return 2
            else:
                return 3
        if source in ["GallicaPress", "Pile-FreeLaw", "RedPajama"]:
            return 1
        if source in ["AmericanStories", "FineWebEdu"]:
            return 1.5
        if source in ["AmendementsParlement", "Claire", "DiscoursPublics", "Europarl"]:
            return 2
        if source in [
            "GallicaMonographies",
            "HAL",
            "InterventionsParlement",
            "OpenData",
            "LEGI",
        ]:
            return 2
        if source in [
            "OpenEdition",
            "QuestionsEcritesParlement",
            "Stac",
            "TheStack",
            "Theses",
            "YouTube",
        ]:
            return 2
        if source in [
            "Pile-NIH_ExPorter",
            "Pile-PhilPapers",
            "Pile-StackExchange",
            "Pile-Ubuntu_IRC",
            "Pile-USPTO_Backgrounds",
            "PeS2o",
        ]:
            return 2.5
        if source in [
            "Pile-DM_Mathematics",
            "Wikipedia",
            "Wikisource",
            "Wiktionary",
            "CroissantAligned",
            "EuroparlAligned",
        ]:
            return 3
        return 0

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        source = doc.metadata["source"]
        language = doc.metadata["language"]
        repeat = self.get_repeat(source, language)
        threshold = repeat * self.rate
        return self.uniform() < threshold


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subset to load",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision",
        choices=["main", "v1.2"],
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    revision = args.revision

    pipeline = [
        HuggingFaceDatasetReader(
            "OpenLLM-France/Lucie-Training-Dataset",
            {"revision": revision, "split": "train"},
            streaming=True,
        ),
        SamplerFilter(seed=42),
        JsonlWriter(f"{DATA_PATH}/lucie_training_mix/{revision}/data"),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/lucie_training_mix/{revision}/logs",
        job_name="lucie_training_mix",
        cpus_per_task=2,
        tasks=100,
    )

    main_processing_executor.run()
