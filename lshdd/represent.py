"""Vocabulary building and token weighting."""
import numpy as np
from collections import Counter


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


def idf_weights(vocab, gamma=2.0, class_weight=None):
    """w_t = alpha_class(t) * idf_t ** gamma  (tokens present in every product get 0)."""
    w = vocab.idf() ** gamma
    if class_weight:
        cls = vocab.classes()
        mult = np.array([class_weight.get(c, 1.0) for c in cls])
        w = w * mult
    return w


def weighted_jaccard(a, b, w):
    """Exact weighted Jaccard of two sorted id arrays."""
    inter = np.intersect1d(a, b, assume_unique=True)
    wi = w[inter].sum()
    wu = w[a].sum() + w[b].sum() - wi
    return wi / wu if wu > 0 else 0.0
