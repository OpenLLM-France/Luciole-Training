import re
import json
import warnings


def extract_educational_json(
    text: str, keys=["educational_score", "topic", "is_ad", "is_toxic"]
) -> dict:
    pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    matches = pattern.findall(text)

    default_output = {k: None for k in keys}

    if not matches:
        warnings.warn("No JSON match found", UserWarning)
        return default_output

    try:
        data_dict = json.loads(matches[0])
        for k in keys:
            default_output[k] = data_dict.get(k, None)
        return default_output
    except json.JSONDecodeError:
        warnings.warn("Failed to extract JSON", UserWarning)
        return default_output


def extract_text(text: str) -> dict | None:
    pattern = re.compile(r"Web page:\n\n(.*?)\n\n---", re.DOTALL)

    matches = pattern.findall(text)
    match = matches[0]
    try:
        return match
    except json.JSONDecodeError:
        warnings.warn("Failed to extract text", UserWarning)
        return ""


def normalize_text(text: str) -> str:
    text = text.lower()

    # Apply the sequence of substitutions
    text = re.sub(r"'", " ' ", text)
    text = re.sub(r'"', "", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r":", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text
