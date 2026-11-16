"""MinHash and weighted MinHash signatures (own implementation, numpy only).

Tokens are first mapped to 31-bit integers x (a fixed hash of the token string). Hash function
k is the universal hash h_k(x) = (a_k * x + b_k) mod p with p = 2^31 - 1.

Unweighted MinHash:  signature entry k = token minimising h_k.
Weighted MinHash:    every token t gets an "arrival time" e_k(t) = -ln(u_k(t)) / w_t with
                     u_k(t) = (h_k(x_t) + 1) / (p + 1) in (0, 1]; the signature entry is the
                     token with the earliest arrival. The minimum of independent exponentials
                     with rates w_t is attained by token t with probability w_t / sum(w), so two
                     sets A, B agree on entry k with probability

                         sum_{t in A & B} w_t / sum_{t in A | B} w_t ,

                     the weighted Jaccard similarity (Chum et al., BMVC 2008).
                     With w_t = 1 this reduces to ordinary MinHash.
"""
import zlib
import numpy as np

PRIME = (1 << 31) - 1


def token_codes(tokens):
    """Deterministic 31-bit integer code for every token (class, string)."""
    return np.array([zlib.crc32(("%s\x1f%s" % t).encode("utf-8")) % PRIME for t in tokens],
                    dtype=np.int64)


def arrival_times(codes, weights, n_hashes, seed=0):
    """Matrix e[k, t] of arrival times of token t under hash function k."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=n_hashes, dtype=np.int64)
    b = rng.integers(0, PRIME, size=n_hashes, dtype=np.int64)
    h = (a[:, None] * codes[None, :] + b[:, None]) % PRIME        # < 2^62, exact in int64
    e = -np.log((h + 1.0) / (PRIME + 1.0))
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        e = np.where(w[None, :] > 0, e / np.maximum(w, 1e-300)[None, :], np.inf)
    return e


def signatures(token_ids, codes, weights, n_hashes, seed=0):
    """token_ids: list of int arrays indexing the vocabulary (one per product).
    Returns an int64 matrix (n_products, n_hashes) holding the selected token code.
    Products without any positively weighted token get unique negative entries, so they
    never collide with anything."""
    e = arrival_times(codes, weights, n_hashes, seed)
    sig = np.empty((len(token_ids), n_hashes), dtype=np.int64)
    for i, t in enumerate(token_ids):
        if len(t) == 0:
            sig[i] = -(i + 1)
            continue
        et = e[:, t]
        best = np.argmin(et, axis=1)
        if not np.isfinite(et[np.arange(n_hashes), best]).all():
            sig[i] = -(i + 1)
            continue
        sig[i] = codes[t[best]]
    return sig
