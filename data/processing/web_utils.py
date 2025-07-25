from datatrove.pipeline.formatters import PIIFormatter, PhoneNumberPII
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.robots_txt_filter import RobotsTxtFilter
from datatrove.pipeline.filters import LambdaFilter
import os
import json

FASTTEXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"), "fasttext_classifiers/fineweb_edu_annotation"
)
DECONT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"),
    "data/raw_data/full_datasets/decontamination_index/data",
)
ROBOTSTXT_PATH = os.path.join(
    os.getenv("OpenLLM_OUTPUT"),
    "data/raw_data/full_datasets/robots_txt/cc-main-2025-26/data_merge",
)


def get_duplicated_urls(path="duplicated_urls.json"):
    wiki_languages = [
        "ar",
        "br",
        "ca",
        "co",
        "de",
        "en",
        "es",
        "eu",
        "fr",
        "frp",
        "it",
        "nl",
        "oc",
        "pcd",
        "pt",
    ]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_keywords = []
    for sublist in data.values():
        for item in sublist:
            if "{lan}" in item:
                all_keywords.extend(
                    item.replace("{lan}", lang) for lang in wiki_languages
                )
            else:
                all_keywords.append(item)

    return all_keywords


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
        else:
            doc.metadata["edu_score"] = "unk"
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
        ]
    else:
        edu_filters = []
    if not edu_filters:
        print(
            f"Model not found at {model_url}. Skipping educational filters for {language}."
        )
    edu_filters.append(edu_score)
    return edu_filters


def get_decontamination_filters(
    language, output_path, decontamination_path=DECONT_PATH
):
    iso_language = map_language_to_iso.get(language, language)
    index_folder = os.path.join(decontamination_path, iso_language)
    if os.path.exists(index_folder):
        filters = [
            NGramsDecontFilter(
                index_folder=index_folder,
                config=NGramsDecontConfig(
                    n_grams=13, find_query_ngrams=True, find_overlap_ngrams=True
                ),
                language=language,
                exclusion_writer=JsonlWriter(
                    f"{output_path}/removed/decont",
                ),
            )
        ]
    else:
        filters = []
    if not filters:
        print(
            f"Decontamination index not found for {iso_language} at {index_folder}. Skipping decontamination filters."
        )
    return filters


def get_robot_filter(output_path, robots_txt_path=ROBOTSTXT_PATH):
    return RobotsTxtFilter(
        robots_txt_path=robots_txt_path,
        exclusion_writer=JsonlWriter(
            f"{output_path}/removed/robots_txt",
        ),
    )


def get_dedup_filter(output_path):
    def deduplicate_url(doc):
        url = doc.metadata.get("url", None)
        if url is None:
            return True
        for keyword in get_duplicated_urls():
            if url.startswith(keyword):
                return False, f"duplicate_url:{keyword}"
        return True

    return LambdaFilter(
        deduplicate_url,
        exclusion_writer=JsonlWriter(
            f"{output_path}/removed/duplicated_url",
        ),
    )


def get_web_pipeline(
    language,
    output_path,
    do_edu=True,
    do_pii=True,
    do_decont=False,
    robots_txt_path=ROBOTSTXT_PATH,
):
    edu_filters = get_edu_filters(language) if do_edu else []
    pii_formatter = get_pii_formatter(language) if do_pii else []
    decontamination_filters = (
        get_decontamination_filters(language, output_path) if do_decont else []
    )

    pipeline = [
        get_dedup_filter(output_path),
        get_robot_filter(output_path, robots_txt_path=robots_txt_path),
        *edu_filters,
        *pii_formatter,
        *decontamination_filters,
    ]
    return pipeline
