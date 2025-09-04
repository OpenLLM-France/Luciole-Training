import os
from nemo_patch import indexed_dataset
import random


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
        description="Split MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", type=str, help="Input indexed dataset filename (without extension)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output filename prefix (without extension neither split number)",
    )
    parser.add_argument(
        "ratio", type=float, help="Split ratios (ex: 0.6 0.4)", nargs="+"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Vocabulary size (is it larger or not than 65500?)",
    )
    args = parser.parse_args()

    input = args.input
    output = args.output
    vocab_size = args.vocab_size

    assert input
    assert output and output != input
    assert os.path.exists(input + ".bin"), "Input .bin file must exist."
    assert os.path.exists(input + ".idx"), "Input .idx file must exist."

    output_bin_files = [output + f"_{i}.bin" for i in range(len(args.ratio))]
    output_idx_files = [output + f"_{i}.idx" for i in range(len(args.ratio))]

    if not os.path.isdir(os.path.dirname(output_bin_files[0])):
        os.makedirs(os.path.dirname(output_bin_files[0]))

    # Aggregate data and write output bin
    builders = [
        indexed_dataset.make_builder(
            output_bin_file, impl="mmap", vocab_size=vocab_size
        )
        for output_bin_file in output_bin_files
    ]

    try:
        num_tokens = 0
        dataset = indexed_dataset.MMapIndexedDataset(input)
        for doc in dataset:
            num_tokens += len(doc)
            if num_tokens > args.max_tokens:
                break
            r = random.random()
            cumulative = 0.0
            for i in range(len(args.ratio)):
                if cumulative + args.ratio[i] > r:
                    builders[i].add_doc(doc, [len(doc)])
                    break
                cumulative += args.ratio[i]
    except (Exception, KeyboardInterrupt) as err:
        for output_bin_file in output_bin_files:
            if os.path.exists(output_bin_file):
                os.remove(output_bin_file)
        raise err

    for builder, output_idx_file in zip(builders, output_idx_files):
        builder.finalize(output_idx_file)
