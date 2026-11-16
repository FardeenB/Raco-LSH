"""Sanity checks and timings.

1. Leakage check: token sets are identical when the gold model IDs are removed from the
   product records (the experiments pass records without the gold field anyway).
2. Robustness: F1 at k=3 on a second, independent set of bootstrap splits.
3. Wall-clock time of the LSH pipeline versus scoring all cross-shop pairs, on the test
   collections and on the complete data set.

usage: python experiments/run_checks.py [n_bootstraps]
"""
import sys, time
from itertools import combinations
from common import *
from lshdd.evaluate import bootstrap_splits
from lshdd.pipeline import DuplicateDetector, featurize
from lshdd.tokens import product_tokens


def exhaustive_seconds(det, prods):
    t0 = time.perf_counter()
    idx = det.blocker.index(prods)
    sh = [p["shop"] for p in prods]
    allp = [(i, j) for i, j in combinations(range(len(prods)), 2) if sh[i] != sh[j]]
    det.matcher.predict(featurize(idx, allp))
    return time.perf_counter() - t0, len(allp)


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    P = load_products(DATA)
    same = all(product_tokens(p) == product_tokens(public(p)) for p in P)
    print("tokens identical without gold field:", same)
    rows = []
    for bi, (tr, te) in enumerate(splits(len(P), nb)):
        ptr, pte = [public(P[i]) for i in tr], [public(P[i]) for i in te]
        g_tr, g_te = local_gold(P, tr), local_gold(P, te)
        t0 = time.perf_counter()
        det = DuplicateDetector(k=3, seed=500 + bi).fit(ptr, g_tr)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred, cands = det.predict(pte)
        t_pred = time.perf_counter() - t0
        t_all, n_all = exhaustive_seconds(det, pte)
        allprods = [public(p) for p in P]
        t0 = time.perf_counter()
        _, cands_full = det.predict(allprods)
        t_full = time.perf_counter() - t0
        t_full_all, n_full_all = exhaustive_seconds(det, allprods) if bi < 3 else (float("nan"), 0)
        rows.append(dict(F1=prf(pred, g_te)["F1"], fit_seconds=t_fit, predict_seconds=t_pred,
                         exhaustive_predict_seconds=t_all, n_candidates=len(cands), n_all_pairs=n_all,
                         n_test=len(te), full_predict_seconds=t_full, full_n_candidates=len(cands_full),
                         full_exhaustive_seconds=t_full_all))
        print({k: round(v, 3) for k, v in rows[-1].items()})
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    alt = []
    for bi, (tr, te) in enumerate(bootstrap_splits(len(P), 30, seed=777)):
        det = DuplicateDetector(k=3, seed=4000 + bi).fit([public(P[i]) for i in tr], local_gold(P, tr))
        pred, _ = det.predict([public(P[i]) for i in te])
        alt.append(prf(pred, local_gold(P, te))["F1"])
    out["F1_alternative_splits_mean"] = float(np.mean(alt))
    out["F1_alternative_splits_sd"] = float(np.std(alt))
    out["F1_alternative_splits_n"] = len(alt)
    out["tokens_identical_without_gold"] = bool(same)
    out["n_boot"] = len(rows)
    save("checks.json", out)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()})
