"""The compiled clustering (msm._merge_kernel) against the Python reference (msm._merges) on
random collections, for both ways of deciding which pairs are at distance infinity.
usage: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lshdd import msm

pytestmark = pytest.mark.skipif(msm.njit is None, reason="numba not installed")
EPS = [round(0.1 * i, 1) for i in range(1, 10)]


def collection(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 120))
    shop = rng.integers(0, 4, n).tolist()
    model = rng.integers(0, max(2, n // 2), n)          # gold: products of one model
    gold = {(a, b) for a in range(n) for b in range(a + 1, n) if model[a] == model[b]}
    allp = [(a, b) for a in range(n) for b in range(a + 1, n)]
    keep = rng.random(len(allp)) < rng.uniform(0.05, 0.6)
    pairs = [p for p, k in zip(allp, keep) if k]
    dist = np.round(rng.random(len(pairs)), 2)          # rounding creates ties
    return n, shop, gold, pairs, dist


@pytest.mark.parametrize("seed", range(60))
def test_same_shop_rule(seed):
    n, shop, gold, pairs, dist = collection(seed)
    fast = msm.same_shop(shop)
    python = lambda a, b: shop[a] == shop[b]           # a plain predicate takes the Python path
    assert msm.f1_path(n, pairs, dist, gold, EPS, fast) == msm.f1_path(n, pairs, dist, gold, EPS, python)
    assert msm.cluster(n, pairs, dist, 0.5, fast) == msm.cluster(n, pairs, dist, 0.5, python)


@pytest.mark.parametrize("seed", range(60))
def test_unscored_pairs_rule(seed):
    n, shop, gold, pairs, dist = collection(seed)
    finite = set(pairs)
    python = lambda a, b: ((a, b) if a < b else (b, a)) not in finite
    assert msm.f1_path(n, pairs, dist, gold, EPS) == msm.f1_path(n, pairs, dist, gold, EPS, python)
    assert msm.cluster(n, pairs, dist, 0.5) == msm.cluster(n, pairs, dist, 0.5, python)


def test_gold_that_is_not_a_clustering_falls_back():
    # (0, 1) and (1, 2) without (0, 2): no labels exist, the Python path is used.
    # eps 0.15: {0, 1}, 1 of 1 predicted pairs right -> 2 / (1 + 2); eps 0.25: {0, 1, 2}, 2 of 3 -> 4 / 5
    assert msm._gold_labels(3, {(0, 1), (1, 2)}) is None
    pairs, d = [(0, 1), (1, 2), (0, 2)], np.array([0.1, 0.2, 0.3])
    assert msm.f1_path(3, pairs, d, {(0, 1), (1, 2)}, [0.15, 0.25, 0.35]) == pytest.approx([2 / 3, 0.8, 0.8])
