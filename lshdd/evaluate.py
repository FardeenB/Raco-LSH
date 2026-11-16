"""Bootstrap protocol and evaluation measures."""
import numpy as np
from .data import gold_pairs


def bootstrap_splits(n, n_boot, seed=0):
    """Yield (train_idx, test_idx): train = distinct drawn instances (~63%), test = out-of-bag."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        train = np.unique(draw)
        test = np.setdiff1d(np.arange(n), train)
        yield train, test


def blocking_scores(cands, gold, n):
    """cands, gold: sets of local pairs (i<j). Returns PQ, PC, F1*, fraction of comparisons."""
    found = len(cands & gold)
    nc = len(cands)
    pq = found / nc if nc else 0.0
    pc = found / len(gold) if gold else 0.0
    f1s = 2 * pq * pc / (pq + pc) if pq + pc > 0 else 0.0
    frac = nc / (n * (n - 1) / 2)
    return dict(PQ=pq, PC=pc, F1star=f1s, frac=frac, n_cand=nc)


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return dict(precision=p, recall=r, F1=f)


def local_gold(products, idx):
    """Gold pairs expressed in positions 0..len(idx)-1 of the subset."""
    pos = {int(g): k for k, g in enumerate(idx)}
    return {(pos[a], pos[b]) for a, b in gold_pairs(products, [int(i) for i in idx])}
