"""Vocabulary building and token weighting."""
import numpy as np


class Vocabulary:
    def __init__(self, token_sets):
        self.index = {}
        self.ids = []
        for s in token_sets:
            arr = np.fromiter((self.index.setdefault(t, len(self.index)) for t in sorted(s)),
                              dtype=np.int64, count=len(s))
            self.ids.append(arr)
        self.tokens = [None] * len(self.index)
        for t, i in self.index.items():
            self.tokens[i] = t
        self.df = np.zeros(len(self.index))
        for arr in self.ids:
            self.df[arr] += 1
        self.n = len(token_sets)

    def idf(self):
        return np.log(self.n / self.df)

    def classes(self):
        return np.array([t[0] for t in self.tokens])


def idf_weights(vocab, gamma=2.0):
    """w_t = idf_t ** gamma  (tokens present in every product get 0)."""
    return vocab.idf() ** gamma

