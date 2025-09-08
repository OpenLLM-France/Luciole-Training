import unicodedata
import pandas as pd
import random
import tqdm


def get_unicode_chars(start, end):
    return [
        chr(code_point)
        for code_point in range(start, end + 1)
        # if unicodedata.category(chr(code_point))[0] in 'L'  # Only letters
    ]


# Basic Latin (U+0000–U+007F)
latin_basic = get_unicode_chars(0x0041, 0x007A)

# Latin-1 Supplement (U+0080–U+00FF)
latin1 = get_unicode_chars(0x00C0, 0x00FF)

# Latin Extended-A (U+0100–U+017F)
latin_ext_a = get_unicode_chars(0x0100, 0x017F)

# Latin Extended-B (U+0180–U+024F)
latin_ext_b = get_unicode_chars(0x0180, 0x024F)

latin_chars = latin_basic + latin1 + latin_ext_a + latin_ext_b

# Greek and Coptic (U+0370–U+03FF)
greek_chars = get_unicode_chars(0x0370, 0x03FF)

# Arabic (U+0600–U+06FF)
arabic_chars = get_unicode_chars(0x0600, 0x06FF)

# Arabic Supplement (U+0750–U+077F)
arabic_supplement = get_unicode_chars(0x0750, 0x077F)

# Arabic Extended-A (U+08A0–U+08FF)
arabic_ext_a = get_unicode_chars(0x08A0, 0x08FF)

arabic_all = arabic_chars + arabic_supplement + arabic_ext_a

code_symbols = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "(){}[];,.:+-*/%=<>&|^!~#@?_\\'\"` \t\n"
)


def count_characters(dataset, filename):
    alphabet = {}
    for i, text in enumerate(dataset):
        text = text["text"]
        for char in text:
            if char not in alphabet:
                alphabet[char] = 1
            else:
                alphabet[char] += 1

        if i % 10000 == 0:
            dump_alphabet(alphabet, filename)

    dump_alphabet(alphabet, filename)


def dump_alphabet(alphabet, filename):
    alphabet = sorted(alphabet.items(), key=lambda item: item[1], reverse=True)
    with open(filename, "w", encoding="utf-8") as f:
        for char, count in alphabet:
            o = ord(char)
            cat = unicodedata.category(char)
            if char in latin_chars:
                family = "LATIN"
            elif char in greek_chars:
                family = "GREEK"
            elif char in arabic_all:
                family = "ARABIC"
            elif char in code_symbols:
                family = "CODE"
            else:
                family = "OTHER"
            char = (
                char.replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\r", "\\r")
                .replace(" ", "▁")
            )
            f.write(f"{char}\t{count}\t{o}\t{cat}\t{family}\n")


def my_parquet_dataset(parquets):
    random.shuffle(parquets)

    for parquet in tqdm.tqdm(parquets):
        df = pd.read_parquet(parquet)
        for _, row in df.iterrows():
            yield {"text": row["text"]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare alphabet from dataset.")
    parser.add_argument(
        "parquet", type=str, help="Path to the dataset file.", nargs="+"
    )
    args = parser.parse_args()

    if len(args.parquet) == 1 and args.parquet[0].endswith(".tsv"):
        # Rewrite TSV file
        with open(args.parquet[0], "r", encoding="utf-8") as f:
            alphabet = {}
            for iline, line in enumerate(f):
                line = line.rstrip()
                row = line.split("\t")
                if not len(row):
                    continue
                try:
                    alphabet[chr(int(row[2]))] = int(row[1])
                except Exception as e:
                    raise ValueError(
                        f"Error processing line {iline}: '{line}' {len(row)=}"
                    ) from e
        dump_alphabet(alphabet, "alphabet.tsv")

    else:
        ds = my_parquet_dataset(args.parquet)
        count_characters(ds, "alphabet.tsv")
