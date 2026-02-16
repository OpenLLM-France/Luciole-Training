import transformers
from tokenizer_train import test_tokenizer


def norm_for_display(s):
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def norm_tokens(s):
    if isinstance(s, str):
        return norm_for_display(s.replace("Ċ", "\n").replace("ĉ", "\t").replace("<0x0A>", "\n").replace("<0x09>", "\t"))
    return [norm_tokens(t) for t in s]


def norm_spaces(s):
    return s.replace("\u00A0", " ").replace("\r", "")


if __name__ == "__main__":
    example_tokenizers = [
        "gpt-3.5-turbo",
        "gpt-4",
        "bigscience/bloom-7b1",
        "google/gemma-7b",
        "tiiuae/falcon-7b",
        "allenai/OLMo-7B",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "croissantllm/CroissantLLMBase",
        "CohereForAI/c4ai-command-r-plus",
        # "Lucie2.9",
    ]

    import argparse

    parser = argparse.ArgumentParser(description="Test a tokenizer.")
    parser.add_argument(
        "tokenizer",
        help="Tokenizer to evaluate. Examples: "
        + ", ".join(example_tokenizers)
        + "... or a folder containing a tokenizer.",
        nargs="?",
    )
    parser.add_argument(
        "sentence",
        help="A sentence to test on.",
        nargs="*",
    )
    args = parser.parse_args()

    example_sentences = [
        # # Corner case characters
        # "\r\x00",
        # # Unknown characters
        # "␣ あ゙ 😀😘💀",
        # # Digits
        # "$1 234\u00A0567,890  00€ 1.2 3/4",
        # "3² ½ ñ Ω À ẹ́ בּ",
        # # Spaces
        # "Hello Hello\nHello \n Hello   \n   Hello\n\nHello\n\n\n\n\n\n\nHello",
        # # Spaces & Brackets
        # "(Hello) (Hello)\n(Hello) \n (Hello)   \n" + " " * 13 + "(Hello)",
        # "'Hello' 'Hello'\n'Hello' \n 'Hello'   \n" + " " * 13 + "'Hello' «children» children. «children.» «children».",
        # "[INST] [Mr. le président:] [M. Correspondent:]",
        # # Punctuations
        # "website.fr www.com es. As. 1s. (etc...)",
        # # Equations
        # "a.(b+c)-d÷e×f belle-mère grand-mother",
        # "Un abat-jour.\nUn Un    Un\tUn\t\tUn 😀\n\na.(b+c)÷e×f $1\u00A0234,567.89 ½あ゙",
        "I'm it's s'appelle belle-mère aujourd'hui z'a - it’s s’appelle aujourd’hui z’a. 30°C   maison   12\n\n\n\t   ",
    ]

    if args.sentence:
        example_sentences = [" ".join(args.sentence).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")]

    tokenizers = [args.tokenizer] if args.tokenizer else example_tokenizers

    short_output = not args.tokenizer

    if short_output:
        example_sentences = [example_sentences[-1]]

    for tokenizer_name in tokenizers:
        if tokenizer_name.lower().startswith("gpt"):
            import tiktoken

            tokenizer = tiktoken.encoding_for_model(tokenizer_name.lower())
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        for example_sentence in example_sentences:
            tokens, decoded = test_tokenizer(tokenizer, example_sentence)

            # print("* Decoded:", norm_for_display(decoded))
            has_bos = decoded.startswith("<")
            has_eos = decoded.endswith(">")
            if has_bos:
                decoded = decoded[decoded.index(">") + 1 :]
            if has_eos:
                while len(decoded) and decoded[-1] != "<":
                    decoded = decoded[:-1]
                decoded = decoded[:-1]
            bos_eos_details = (
                f"{' no 'if has_bos or has_eos else '....'}{'BOS' if has_bos else '...'}{'/EOS' if has_eos else '....'}"
            )
            if short_output:
                print(tokenizer_name)
                print(" | ".join(norm_tokens(tokens)))
            else:
                print("-" * 50)
                print("* Reference.........:", norm_for_display(example_sentence))
                print(
                    f"* Decoded{bos_eos_details}:",
                    norm_for_display(decoded),
                )
                print("* Tokens: ", norm_tokens(tokens))

            decoded = decoded.lstrip()
            example_sentence = example_sentence.lstrip()
            # print("* OK 100%.....................:", example_sentence == decoded)
            example_sentence = norm_spaces(example_sentence)
            decoded = norm_spaces(decoded)
            # print("* OK up to space normalization:", example_sentence == decoded)
            if example_sentence != decoded:
                print("KO!!!")
                for i, (a, b) in enumerate(zip(example_sentence, decoded)):
                    if a != b:
                        print(f"  {i}: '{a}' ({ord(a)}) != '{b}' ({ord(b)})")
                        break
