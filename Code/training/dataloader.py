import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
from typing import List

from data_util import read_vocab


class BibleData(Dataset):
    def __init__(self, corpus_fn: str, PAD: str="<P>", START: str="<S>", END: str="<E>", UNK: str = "<UNK>"):
        self.PAD = PAD
        self.START = START
        self.END = END
        self.UNK = UNK
        # Get a List of unique words in the corpus
        self.vocab = read_vocab(corpus_fn)
        self.char2i, self.i2char = self._make_char_index(self.vocab)

    def _make_char_index(self, vocab: List):
        """Generate Dicts for encoding/decoding chars as unique indices"""
        chars = set()
        for word in vocab:
            [chars.add(c) for c in word]

        char2i = {c: i for i, c in enumerate(sorted(chars))}
        # Index special symbols at the end
        for special in [self.PAD, self.START, self.END]:
            char2i[special] = len(char2i)
        # Set indexing by i for lookup (this could also just be a List..)
        i2char = {i: c for c, i in char2i.items()}

        return char2i, i2char

    def encode(self, word: str, add_start_tag: bool=True, add_end_tag: bool=False):
        """Encode a sequence as a pytorch Tensor of indices"""
        seq = [self.char2i[self.START]] if add_start_tag else []
        seq.extend([self.char2i.get(c, self.UNK) for c in word])
        seq = seq + [self.char2i[self.END]] if add_end_tag else seq

        return torch.LongTensor(seq)

    def decode(self, word: torch.Tensor):
        """Take the tensor of indices, and return a List of chars"""
        return [self.i2char[c] for c in word.numpy()]

    @property
    def pad_idx(self):
        return self.char2i[self.PAD]

    @property
    def character_size(self):
        """Number of unique characters. This will correspond to the size of the embeddings martix"""
        return len(self.char2i)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, index):
        word = self.vocab[index]
        # start tag to last char
        inp = self.encode(word, add_start_tag=True, add_end_tag=False)
        # First char to end_tag. This offsets target to always be 1 char ahead.
        target = self.encode(word, add_start_tag=False, add_end_tag=True)

        return inp, target


class ReverseBibleData(BibleData):
    
    def encode(self, word: str, add_start_tag: bool=True, add_end_tag: bool=False):
        """Encode a sequence as a pytorch Tensor of indices"""
        seq = [self.char2i[self.END]] if add_end_tag else []
        seq.extend([self.char2i.get(c, self.UNK) for c in word[::-1]])
        seq = seq + [self.char2i[self.START]] if add_START_tag else seq

        return torch.LongTensor(seq)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def pad_collate(self, batch: List):
        """
        batch: List of batch_size tuples of input/output pairs of tensors.
        return: a tensor of all words in 'batch' after padding
        """
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # seperate input/target seqs
        batch_in, batch_out = zip(*batch)

        # pad according to max_len
        max_len_in = max([len(t) for t in batch_in])
        padded_in = [self.pad_tensor(t, max_len_in) for t in batch_in]
        batch_in = torch.stack(padded_in)

        max_len_out = max([len(t) for t in batch_out])
        padded_out = [self.pad_tensor(t, max_len_out) for t in batch_out]
        batch_out = torch.stack(padded_out)

        return batch_in, batch_out

    def pad_tensor(self, t: torch.Tensor, max: int):
        # Tuple of prepend indices, append_indices
        p = (0, max-len(t))
        # Add p pad_idx to t
        return F.pad(t, p, "constant", self.pad_idx)

    def __call__(self, batch):
        return self.pad_collate(batch)
