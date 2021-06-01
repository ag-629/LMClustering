from typing import List


def read_vocab(fn: str) -> List:
    """Read in the bible text file, returning the unique vocab as a List

    Note we lowercases all words."""
    print(f"Reading corpus {fn.split('/')[-1]}... ")
    vocab = set()
    with open(fn) as f:
        for line in f:
            line = [w.lower() for w in line.strip().split()]
            for w in line:
                vocab.add(w)

    print(f"Found vocab of size {len(vocab)}.\n")

    # Sort to change type to List, and keep order consistent for a given vocab
    return sorted(vocab)
