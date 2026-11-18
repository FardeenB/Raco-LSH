"""Re-implementations of the product representations of MSMP and MSMP+ (LSH part only).

MSMP  (van Dam et al., SAC 2016):       binary vector over the model words of the title.
MSMP+ (Hartveld et al., CAiSE 2018):    data cleaning (inch / hz), title model words, title model
                                        words that re-occur in attribute values, and numeric
                                        model words of the attribute values (unit stripped).
Both use ordinary (unweighted) MinHash with a signature of 50% of the vector length and the
classical banding technique.

lsh_neighbours() is the pre-selection of the MSMP experiments (run_msm.py). The signature length
n and a threshold t fix the bands: the b whose (1/b)^(b/n) is closest to t, r = n // b. With
code=True it also follows the MSMP authors' code (students/group6 in
github.com/nickvalstar/APFAProject: ModelWordSplitter, LSH/MinHash, LSH/LocalitySensitiveHash,
LSH/HashFunction): MinHash with explicit random permutations, and band keys that concatenate the
signature entries without a separator, so that e.g. [1, 23] and [12, 3] share a bucket.
"""
import re
import numpy as np

from .represent import Vocabulary
from .minhash import signatures, token_codes
from .lsh import candidate_pairs

_MW_TITLE = re.compile(r'[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*')
# model-word regex as written in the MSM authors' Java code: "^" inside the class is a literal
_MW_JAVA = re.compile(r"([a-zA-Z0-9]*(([0-9]+[^0-9^,^ ]+)|([^0-9^,^ ]+[0-9]+))[a-zA-Z0-9]*)")
_MW_VALUE = re.compile(r'^\d+(?:\.\d+)?[a-zA-Z]+$|^\d+(?:\.\d+)?$')
_NUM = re.compile(r'^\d+(?:\.\d+)?')

_INCH = re.compile(r'\s*(?:-\s*)?(?:inches|inch|"|”)', re.I)
_HZ = re.compile(r'\s*-?\s*(?:hertz|hz)', re.I)


def msmp_tokens(p):
    return {("mw", m) for m in _MW_TITLE.findall(p["title"])}


def msmp_tokens_code(p):
    """Title model words as group6/ModelWordSplitter.java: Java regex on the raw title, words of
    more than one character."""
    return {("mw", m.group(1)) for m in _MW_JAVA.finditer(p["title"]) if len(m.group(1)) > 1}


def _clean(s):
    return _HZ.sub("hz", _INCH.sub("inch", s)).lower()


def msmp_plus_tokenizer(products, decimals_only=True, title_mw_in_values=True):
    """MSMP+ needs the title model words of the whole collection, so it is a factory.

    decimals_only=True keeps only value numbers that contain a decimal point ("we only add
    decimal numbers"); False follows the published regular expression literally, which also
    admits integers. We report the better of the two variants for the baseline."""
    title_mw = [set(_MW_TITLE.findall(_clean(p["title"]))) for p in products]
    all_title_mw = set().union(*title_mw) if title_mw else set()
    cache = {}
    for p, tm in zip(products, title_mw):
        out = {("mw", m) for m in tm}
        for v in p["features"].values():
            for tok in _clean(v).split():
                if title_mw_in_values and tok in all_title_mw:
                    out.add(("mw", tok))
                if _MW_VALUE.match(tok):
                    num = _NUM.match(tok).group(0)
                    if "." in num or not decimals_only:
                        out.add(("num", num))
        cache[id(p)] = out
    return lambda p: cache[id(p)]


def classical_candidates(products, tokenizer, rows, seed):
    """Candidate pairs of the MSMP / MSMP+ pre-selection for every r in `rows` (b = n // r bands);
    pairs within one Web shop are skipped."""
    toks = [tokenizer(p) for p in products]
    vocab = Vocabulary(toks)
    n = max(2, len(vocab.tokens) // 2)
    sig = signatures(vocab.ids, token_codes(vocab.tokens), None, n, seed=seed)
    shops = np.array([p["shop"] for p in products])
    return {r: candidate_pairs(sig, n // r, r, block=shops) for r in rows if n // r >= 1}


def bands_for(n, t):
    """(b, r) for signature length n and threshold t: the first b in 1..n whose (1/b)^(b/n) is
    closest to t, r = n // b (LSH/LocalitySensitiveHash.java)."""
    b = np.arange(1, n + 1)
    b = int(b[np.argmin(np.abs((1.0 / b) ** (b / n) - t))])
    return b, n // b


def permutation_signatures(token_ids, n_tokens, n, seed):
    """MinHash with explicit random permutations (LSH/MinHash.java): entry k is the position, in
    permutation k, of the first of the product's tokens. Products without tokens keep the 0 the
    Java arrays start with."""
    rng = np.random.default_rng(seed)
    pos = rng.permuted(np.tile(np.arange(n_tokens, dtype=np.int64), (n, 1)), axis=1)
    sig = np.zeros((len(token_ids), n), dtype=np.int64)
    for i, t in enumerate(token_ids):
        if len(t):
            sig[i] = pos[:, t].min(axis=1)
    return sig


def band_labels(sig, lo, r, concat):
    """Bucket of every product in the band of rows lo..lo+r-1. concat=True keys the bucket by the
    entries written one after the other (HashFunction.concatenate), so different signatures can
    share a bucket."""
    if concat:
        keys = sig[:, lo].astype(str)
        for j in range(lo + 1, lo + r):
            keys = np.char.add(keys, sig[:, j].astype(str))
        return np.unique(keys, return_inverse=True)[1].ravel()
    return np.unique(sig[:, lo:lo + r], axis=0, return_inverse=True)[1].ravel()


def lsh_neighbours(products, tokenizer, ts, seed, code=False):
    """{t: (b, r, N)} for every threshold t in ts, where N[i, j] is True when products i and j share
    a bucket in at least one band (N[i, i] is True). Pairs of one Web shop are included, as in the
    LSH of the papers. Signature length n = 50% of the vector length; code=True uses
    permutation_signatures and concatenated band keys, otherwise lshdd.minhash and exact keys."""
    vocab = Vocabulary([tokenizer(p) for p in products])
    n = max(2, len(vocab.tokens) // 2)
    if code:
        sig = permutation_signatures(vocab.ids, len(vocab.tokens), n, seed)
    else:
        sig = signatures(vocab.ids, token_codes(vocab.tokens), None, n, seed=seed)
    out = {}
    for t in ts:
        b, r = bands_for(n, t)
        N = np.eye(len(products), dtype=bool)
        for k in range(b):
            lab = band_labels(sig, k * r, r, concat=code)
            order = np.argsort(lab, kind="stable")
            for grp in np.split(order, np.flatnonzero(np.diff(lab[order])) + 1):
                if len(grp) > 1:
                    N[np.ix_(grp, grp)] = True
        out[t] = (b, r, N)
    return out


def two_hop(N):
    """Pairs the MSMP code scores: every pair inside the block of one product and its neighbours
    (Group6BlockingScheme with MsmBasedAlgorithm.computeSimilarities), i.e. pairs with a common
    neighbour. N has a True diagonal, so neighbours themselves are included."""
    f = N.astype(np.float32)
    return (f @ f) > 0
