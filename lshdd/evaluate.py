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


def cluster_bootstrap_splits(labels, n_boot, seed=0):
    """Bootstrap over the duplicate clusters (products with equal label) as in sparker, the
    framework of the MSM authors (SamplerImpl in eval/Sampler.scala of
    github.com/nickvalstar/APFAProject): as many clusters as there are are drawn with replacement
    from one pool; train = the drawn clusters, test = the clusters never drawn. Yields
    (train_idx, test_idx) like bootstrap_splits; a cluster is never split over train and test."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    groups = list(groups.values())
    for _ in range(n_boot):
        drawn = np.unique(rng.integers(0, len(groups), size=len(groups)))
        train = np.array(sorted(i for d in drawn for i in groups[d]), dtype=np.int64)
        yield train, np.setdiff1d(np.arange(len(labels)), train)


def group_bootstrap_splits(labels, n_boot, seed=0):
    """Bootstrap over the duplicate groups (products with equal label), as in MSM (van Bezu et al.,
    SAC 2015): the ratio of duplicates versus non-duplicates is maintained by drawing the singleton
    groups and the groups with duplicates separately. Yields (train_idx, test_idx) like
    bootstrap_splits; a group is never split over train and test."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    strata = [[g for g in groups.values() if len(g) == 1], [g for g in groups.values() if len(g) > 1]]
    for _ in range(n_boot):
        train = []
        for st in strata:
            for d in np.unique(rng.integers(0, len(st), size=len(st))):
                train += st[d]
        train = np.array(sorted(train), dtype=np.int64)
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
