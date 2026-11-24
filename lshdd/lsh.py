"""Locality-sensitive hashing by the banding technique (own implementation)."""
from collections import defaultdict, Counter
from itertools import combinations


def band_buckets(sig, bands, rows):
    """Yield, for every band, a dict bucket-key -> list of product positions."""
    n, n_hashes = sig.shape
    assert bands * rows <= n_hashes, "signature too short for this (bands, rows)"
    for b in range(bands):
        chunk = sig[:, b * rows:(b + 1) * rows]
        buckets = defaultdict(list)
        for i in range(n):
            buckets[chunk[i].tobytes()].append(i)
        yield buckets


def candidate_pairs(sig, bands, rows, block=None):
    """Pairs (i, j), i<j, that share at least one band (OR over bands, AND within a band).

    block: optional array; pairs whose entries are equal are skipped while the buckets are
    enumerated (used for the 'no duplicates inside one Web shop' rule, at no extra cost)."""
    cands = set()
    for buckets in band_buckets(sig, bands, rows):
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i, j in combinations(members, 2):
                if block is not None and block[i] == block[j]:
                    continue
                cands.add((i, j))
    return cands


def collision_counts(sig, bands, rows, block=None):
    """Number of bands in which each pair shares a bucket (only pairs with >= 1 collision).

    This is the hash-table side of the (b, r, k) scheme: a pair becomes a candidate when it
    collides in at least k of the b bands. k = 1 is the classical banding technique."""
    counts = Counter()
    for buckets in band_buckets(sig, bands, rows):
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i, j in combinations(members, 2):
                if block is not None and block[i] == block[j]:
                    continue
                counts[(i, j)] += 1
    return counts

