"""Re-implementations of the product representations of MSMP and MSMP+ (LSH part only).

MSMP  (van Dam et al., SAC 2016):       binary vector over the model words of the title.
MSMP+ (Hartveld et al., CAiSE 2018):    data cleaning (inch / hz), title model words, title model
                                        words that re-occur in attribute values, and numeric
                                        model words of the attribute values (unit stripped).
Both use ordinary (unweighted) MinHash with a signature of 50% of the vector length and the
classical banding technique.
"""
import re
import numpy as np

from .represent import Vocabulary
from .minhash import signatures, token_codes
from .lsh import candidate_pairs

_MW_TITLE = re.compile(r'[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*')
_MW_VALUE = re.compile(r'^\d+(?:\.\d+)?[a-zA-Z]+$|^\d+(?:\.\d+)?$')
_NUM = re.compile(r'^\d+(?:\.\d+)?')

_INCH = re.compile(r'\s*(?:-\s*)?(?:inches|inch|"|”)', re.I)
_HZ = re.compile(r'\s*-?\s*(?:hertz|hz)', re.I)


def msmp_tokens(p):
    return {("mw", m) for m in _MW_TITLE.findall(p["title"])}


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
