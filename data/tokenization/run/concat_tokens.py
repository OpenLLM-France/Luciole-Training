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
        description="Concatenate MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input indexed dataset filenames (with extension .bin or without extension)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output filenames prefix (with extension .bin or without extension)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Vocabulary size (is it larger or not than 65500?)",
    )
    args = parser.parse_args()

    inputs = args.inputs
    output = args.output
    vocab_size = args.vocab_size

    assert len(inputs)
    assert output
    assert output not in inputs
    assert all(
        os.path.exists(input_file + ".bin") for input_file in inputs
    ), "All input .bin files must exist."
    assert all(
        os.path.exists(input_file + ".idx") for input_file in inputs
    ), "All input .idx files must exist."

    output_bin_file = output + ".bin"
    output_idx_file = output + ".idx"

    if not os.path.isdir(os.path.dirname(output_bin_file)):
        os.makedirs(os.path.dirname(output_bin_file))

    # Aggregate data and write output bin
    builder = indexed_dataset.make_builder(
        output_bin_file, impl="mmap", vocab_size=vocab_size
    )

    try:
        for path in inputs:
            builder.merge_file_(path)
            # dataset = indexed_dataset.MMapIndexedDataset(path)
            # for doc in dataset:
            #     builder.add_doc(doc, [len(doc)])
    except (Exception, KeyboardInterrupt) as err:
        if os.path.exists(output_bin_file):
            os.remove(output_bin_file)
        raise err

    builder.finalize(output_idx_file)
