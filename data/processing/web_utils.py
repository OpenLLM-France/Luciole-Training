from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontFilter
import os

FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)
DECONT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"),
    "data/raw_data/full_datasets/decontamination_index/data",
)


def edu_score(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    """
    `data` is a generator of Document. You must also return a generator of Document (yield)
    You can optionally use `rank` and `world_size` for sharding
    """
    for doc in data:
        # Handle educational score if present
        edu_score = doc.metadata.pop("edu_score", None)
        if edu_score is not None:
            edu_score_mean = sum(
                int(label.split("__label__")[-1]) * prob
                for label, prob in edu_score.items()
            )
            doc.metadata["edu_score_mean"] = edu_score_mean
            doc.metadata["edu_score"] = int(round(edu_score_mean))
        yield doc


def get_pii_formatter(language):
    pii_cleaning = [
        PIIFormatter(ip_replacement="<IP_ADDRESS>"),
    ]
    if language in ["fra_Latn", "fr"]:
        pii_cleaning.append(
            PhoneNumberPII(["ZZ", "FR", "CA", "BE"], replacement="<PHONE_NUMBER>")
        )
    elif language in ["deu_Latn", "de"]:
        pii_cleaning.append(PhoneNumberPII(["ZZ", "DE"], replacement="<PHONE_NUMBER>"))
    elif language in ["spa_Latn", "es", "cat_Latn", "ca"]:
        pii_cleaning.append(PhoneNumberPII(["ZZ", "ES"], replacement="<PHONE_NUMBER>"))
    elif language in ["ita_Latn", "it"]:
        pii_cleaning.append(PhoneNumberPII(["ZZ", "IT"], replacement="<PHONE_NUMBER>"))
    elif language in ["por_Latn", "pt"]:
        pii_cleaning.append(PhoneNumberPII(["ZZ", "PT"], replacement="<PHONE_NUMBER>"))
    elif language in ["nld_Latn", "nl"]:
        pii_cleaning.append(PhoneNumberPII(["ZZ", "NL"], replacement="<PHONE_NUMBER>"))
    else:
        pii_cleaning.append(PhoneNumberPII(["ZZ"], replacement="<PHONE_NUMBER>"))
    return pii_cleaning


map_language_to_iso = {
    "fra_Latn": "fr",
    "ita_Latn": "it",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "nld_Latn": "nl",
    "por_Latn": "pt",
    "arb_Arab": "ar",
}
map_iso_to_language = {v: k for k, v in map_language_to_iso.items()}


def get_edu_filters(language, fasttext_path=FASTTEXT_PATH):
    language = map_iso_to_language.get(language, language)
    model_url = os.path.join(
        fasttext_path,
        f"Qwen3-32B_content_edu_{language}/model/educational_score_ngram2_epoch5_lr0.1.bin",
    )
    if os.path.exists(model_url):
        edu_filters = [
            FastTextClassifierFilter(
                model_url=model_url,
                keep_labels=("0", 0),
                newline_replacement=" ",
                save_labels_in_metadata=True,
                filter_name="edu_score",
            ),
            edu_score,
        ]
        return edu_filters
    print(
        f"Model not found at {model_url}. Skipping educational filters for {language}."
    )
    return []


def get_decontamination_filters(language, decontamination_path=DECONT_PATH):
    iso_language = map_language_to_iso.get(language, language)
    index_folder = os.path.join(decontamination_path, iso_language)
    if os.path.exists(index_folder):
        filters = [
            NGramsDecontFilter(
                index_folder=index_folder,
                config=NGramsDecontConfig(),
                language=language,
            )
        ]
        return filters
    print(
        f"Decontamination index not found for {iso_language} at {index_folder}. Skipping decontamination filters."
    )
    return []
