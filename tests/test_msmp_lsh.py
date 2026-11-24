"""MSMP pre-selection against the MSMP authors' Java code (students/group6 in
github.com/nickvalstar/APFAProject).  usage: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lshdd import baselines as bl


def java_bands(n, t):
    """Constructor of LSH/LocalitySensitiveHash.java, line by line."""
    min_dev, best = float("inf"), 1
    for b in range(1, n + 1):
        tt = (1.0 / b) ** (b / n)
        if abs(tt - t) < min_dev:
            min_dev, best = abs(tt - t), b
    return best, n // best


def test_bands_for():
    # (1/22)^(22/100) = 0.507 is closest to 0.5 (b = 21: 0.528, b = 23: 0.486)
    assert bl.bands_for(100, 0.5) == (22, 4)
    for n in (7, 50, 451, 670):
        for t in np.arange(0.05, 0.96, 0.05):
            assert bl.bands_for(n, t) == java_bands(n, t)


def test_concatenated_band_keys_collide():
    sig = np.array([[1, 23], [12, 3], [1, 2]])
    lab = bl.band_labels(sig, 0, 2)                       # "123", "123", "12"
    assert lab[0] == lab[1] != lab[2]


def test_permutation_signatures():
    ids = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([], dtype=np.int64), np.array([3])]
    sig = bl.permutation_signatures(ids, 4, 50, seed=0)
    assert (sig[0] == sig[1]).all()
    assert (sig[2] == 0).all()                            # no tokens: the zeros Java starts with
    assert sig.min() >= 0 and sig.max() < 4               # positions in a permutation of 4 rows


def test_permutation_minhash_estimates_jaccard():
    a, b = np.arange(0, 30), np.arange(10, 40)            # Jaccard 20 / 40
    sig = bl.permutation_signatures([a, b], 40, 4000, seed=1)
    assert np.mean(sig[0] == sig[1]) == pytest.approx(0.5, abs=0.03)


def test_lsh_neighbours_matrix():
    prods = [{"title": t, "shop": s, "features": {}} for t, s in
             (('Samsung UN46ES6580 1080p', "a"), ('Samsung UN46ES6580 1080p', "a"), ("Sony KDL40 720p", "b"))]
    out = bl.lsh_neighbours(prods, bl.msmp_tokens_code, (0.5,), seed=0)
    b, r, N = out[0.5]
    assert N.diagonal().all() and (N == N.T).all()
    assert N[0, 1]                                         # identical titles, also within one shop
    assert not N[0, 2]                                     # no model word in common


def test_two_hop():
    N = np.eye(3, dtype=bool)
    N[0, 1] = N[1, 0] = N[1, 2] = N[2, 1] = True           # 0 - 1 - 2
    S = bl.two_hop(N)
    assert S[0, 2] and S[2, 0]                             # 0 and 2 share the block of 1
    assert (S >= N).all()


def test_msmp_tokens_code_raw_title():
    assert bl.msmp_tokens_code({"title": 'Samsung UN46ES6580 46" 3D TV'}) == {
        ("mw", "UN46ES6580"), ("mw", '46"'), ("mw", "3D")}
