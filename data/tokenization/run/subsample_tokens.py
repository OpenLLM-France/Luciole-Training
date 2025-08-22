import os
from nemo_patch import indexed_dataset

def get_filename_without_extension(filename):
    """
    Returns the filename without its extension.
    """
    if filename.endswith(".bin"):
        filename = filename[:-4]
    return filename

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Subsample MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Input indexed dataset filename (without extension)")
    parser.add_argument("output", type=str, help="Output filename prefix (without extension)")
    parser.add_argument("--max_tokens", type=int, default=5_000_000_000, help="Maximum number of tokens per output file")
    parser.add_argument(
        "--vocab_size", type=int, default=128000, help="Vocabulary size (is it larger or not than 65500?)"
    )
    args = parser.parse_args()

    input = args.input
    output = args.output
    vocab_size = args.vocab_size

    assert input
    assert output and output != input
    assert os.path.exists(input + ".bin"), "Input .bin file must exist."
    assert os.path.exists(input + ".idx"), "Input .idx file must exist."

    output_bin_file = output + ".bin"
    output_idx_file = output + ".idx"

    if not os.path.isdir(os.path.dirname(output_bin_file)):
        os.makedirs(os.path.dirname(output_bin_file))

    # Aggregate data and write output bin
    builder = indexed_dataset.make_builder(output_bin_file, impl="mmap", vocab_size=vocab_size)

    try:
        num_tokens = 0
        dataset = indexed_dataset.MMapIndexedDataset(input)
        for doc in dataset:
            num_tokens += len(doc)
            if num_tokens > args.max_tokens:
                break
            builder.add_doc(doc, [len(doc)])
    except (Exception, KeyboardInterrupt) as err:
        if os.path.exists(output_bin_file):
            os.remove(output_bin_file)
        raise err

    builder.finalize(output_idx_file)

