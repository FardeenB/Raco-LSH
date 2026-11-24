"""Bootstrap protocol and evaluation measures."""
import numpy as np
from .data import gold_pairs


def cluster_bootstrap_splits(labels, n_boot, seed=0):
    """Bootstrap over the duplicate clusters (products with equal label) as in sparker, the
    framework of the MSM authors (SamplerImpl in eval/Sampler.scala of
    github.com/nickvalstar/APFAProject): as many clusters as there are are drawn with replacement
    from one pool; train = the products of the drawn clusters (~63% of the clusters), test = the
    out-of-bag clusters. Yields (train_idx, test_idx); a cluster is never split over train and test."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    groups = list(groups.values())
    for _ in range(n_boot):
        drawn = np.unique(rng.integers(0, len(groups), size=len(groups)))
        train = np.array(sorted(i for d in drawn for i in groups[d]), dtype=np.int64)
        yield train, np.setdiff1d(np.arange(len(labels)), train)


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
