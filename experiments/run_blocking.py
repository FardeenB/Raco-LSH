"""Experiment 1: quality of the LSH pre-selection (PC, PQ, F1*) versus fraction of comparisons.

usage: python experiments/run_blocking.py [n_bootstraps]
"""
import sys
from multiprocessing import Pool
from common import *
from lshdd.pipeline import Blocker
from lshdd.tokens import product_tokens, title_tokens, value_tokens
from lshdd.represent import Vocabulary
from lshdd.minhash import signatures, token_codes
from lshdd.lsh import candidate_pairs, collision_counts
from lshdd import baselines

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
N_HASH = 1200


def tok_no_prefix(p):
    return {t for t in product_tokens(p) if t[0] != "ip"}


def tok_no_vq(p):
    return {t for t in product_tokens(p) if t[0] != "vq"}


def tok_title_only(p):
    return {t for t in product_tokens(p) if t[0] in ("tw", "tq", "id")}


OURS = {
    # name: (blocker kwargs, list of rows settings)
    "ours": (dict(gamma=2.0), [1, 2, 3, 4, 6]),
    "ours_gamma0_unweighted": (dict(weighted=False), [3, 4, 6, 8]),
    "ours_gamma1": (dict(gamma=1.0), [3]),
    "ours_gamma3": (dict(gamma=3.0), [3]),
    "ours_no_prefix": (dict(gamma=2.0, tokenizer=tok_no_prefix), [3]),
    "ours_no_quantities": (dict(gamma=2.0, tokenizer=tok_no_vq), [3]),
    "ours_title_words_only": (dict(gamma=2.0, tokenizer=tok_title_only), [3]),
}


def classical_lsh(products, tokenizer, gold, seed):
    """MSMP / MSMP+ style: unweighted MinHash, n = 50% of the vector length, classical banding."""
    cands = baselines.classical_candidates(products, tokenizer, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20], seed)
    return {"r%d" % r: blocking_scores(c, gold, len(products)) for r, c in cands.items()}


def one_bootstrap(args):
    bi, test = args
    P = load_products(DATA)
    prods = [public(P[i]) for i in test]
    gold = local_gold(P, test)
    n = len(prods)
    res = {}
    for name, (kw, rows_list) in OURS.items():
        for r in rows_list:
            blk = Blocker(n_hashes=N_HASH, rows=r, seed=1000 + bi, **kw)
            idx = blk.index(prods)
            cnt = np.array(list(idx["counts"].values()))
            pairs = list(idx["counts"].keys())
            for k in KS:
                if k > blk.bands:
                    continue
                c = {p for p, v in zip(pairs, cnt) if v >= k}
                res["%s|r%d|k%d" % (name, r, k)] = blocking_scores(c, gold, n)
    for key, s in classical_lsh(prods, baselines.msmp_tokens, gold, 1000 + bi).items():
        res["msmp|%s|k1" % key] = s
    tok = baselines.msmp_plus_tokenizer(prods)
    for key, s in classical_lsh(prods, tok, gold, 1000 + bi).items():
        res["msmp_plus|%s|k1" % key] = s
    return res


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    P = load_products(DATA)
    jobs = [(bi, te) for bi, (tr, te) in enumerate(splits(len(P), nb))]
    with Pool(2) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    agg = {}
    for key in allres[0]:
        rows = [r[key] for r in allres if key in r]
        agg[key] = mean_rows(rows)
    save("blocking.json", agg)
    for key in sorted(agg):
        if key.startswith("ours|r3|") or key.startswith("msmp"):
            a = agg[key]
            print("%-28s frac=%.5f PC=%.3f PQ=%.3f F1*=%.3f" % (key, a["frac"], a["PC"], a["PQ"], a["F1star"]))
