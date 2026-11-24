"""Experiment 2: end-to-end duplicate detection (F1) versus fraction of comparisons.

For every bootstrap the matcher is trained on the LSH candidates of the training products and
applied to the LSH candidates of the out-of-bag products. Sweeping k (and r) traces the
efficiency / effectiveness trade-off.

usage: python experiments/run_matching.py [n_bootstraps] [n_processes] [n_bootstraps_exhaustive]
(n_bootstraps_exhaustive: bootstraps that also classify all cross-shop pairs; default all)
"""
import sys, time
from itertools import combinations
from multiprocessing import Pool
from common import *
from lshdd.pipeline import Blocker, Matcher, featurize, _best_threshold

N_HASH = 1200
GRID = [(3, k) for k in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)] + [(2, 2), (2, 1)]


def side(P, ids, r, seed):
    prods = [public(P[i]) for i in ids]
    blk = Blocker(n_hashes=N_HASH, rows=r, seed=seed)
    idx = blk.index(prods)
    return idx


def evaluate(pred, cands, gold, gold_clean, n):
    out = dict(prf(pred, gold))
    c = prf(pred, gold_clean)
    out.update(F1_clean=c["F1"], precision_clean=c["precision"], recall_clean=c["recall"])
    out.update(blocking_scores(set(cands), gold, n))
    return out


def one_bootstrap(args):
    bi, train, test, exhaustive = args
    P = load_products(DATA)
    g_tr, g_te = local_gold(P, train), local_gold(P, test)
    gc_te = cleaned_gold(P, test)
    res = {}
    feats = {}
    for r in sorted({r for r, _ in GRID}):
        itr, ite = side(P, train, r, 500 + bi), side(P, test, r, 900 + bi)
        kmin = min(k for rr, k in GRID if rr == r)
        ptr, pte = Blocker.candidates(itr, kmin), Blocker.candidates(ite, kmin)
        feats[r] = (itr, ite, ptr, featurize(itr, ptr), pte, featurize(ite, pte))
    best_train = (-1.0, None)
    for r, k in GRID:
        itr, ite, ptr, Xtr, pte, Xte = feats[r]
        mtr = np.array([itr["counts"][p] >= k for p in ptr], dtype=bool)
        mte = np.array([ite["counts"][p] >= k for p in pte], dtype=bool)
        ytr = np.array([p in g_tr for p in ptr], dtype=bool)[mtr]
        cte = [p for p, m in zip(pte, mte) if m]
        for model in ("rf", "lr"):
            if model == "lr" and r != 3:
                continue
            m = Matcher(model, seed=bi).fit(Xtr[mtr], ytr, len(g_tr))
            keep = m.predict(Xte[mte])
            pred = {p for p, kp in zip(cte, keep) if kp}
            row = evaluate(pred, cte, g_te, gc_te, len(test))
            res["%s|r%d|k%d" % (model, r, k)] = row
            if model == "rf":
                # model selection on training data only: OOB F1 on the training candidates
                oob = m.clf.oob_decision_function_[:, 1] if m.constant is None else np.zeros(len(ytr))
                tp = int(((oob >= m.threshold) & ytr).sum()); npred = int((oob >= m.threshold).sum())
                f_tr = 2 * tp / (npred + len(g_tr)) if npred + len(g_tr) else 0.0
                if f_tr > best_train[0]:
                    best_train = (f_tr, row)
        # no classifier: threshold on the exact weighted Jaccard, tuned on the training set
        if r == 3:
            thr = _best_threshold(Xtr[mtr][:, 0], ytr, len(g_tr))
            pred = {p for p, x in zip(cte, Xte[mte][:, 0]) if x >= thr}
            res["wj|r%d|k%d" % (r, k)] = evaluate(pred, cte, g_te, gc_te, len(test))
    res["rf|selected_on_train"] = best_train[1]
    if exhaustive:
        t0 = time.perf_counter()
        itr, ite = feats[3][0], feats[3][1]
        def cross(idx):
            sh = [p["shop"] for p in idx["products"]]
            return [(i, j) for i, j in combinations(range(len(sh)), 2) if sh[i] != sh[j]]
        ptr, pte = cross(itr), cross(ite)
        Xtr, Xte = featurize(itr, ptr), featurize(ite, pte)
        ytr = np.array([p in g_tr for p in ptr], dtype=bool)
        m = Matcher("rf", seed=bi).fit(Xtr, ytr, len(g_tr))
        keep = m.predict(Xte)
        pred = {p for p, kp in zip(pte, keep) if kp}
        row = evaluate(pred, pte, g_te, gc_te, len(test))
        row["seconds"] = time.perf_counter() - t0
        res["rf|exhaustive"] = row
    return res


if __name__ == "__main__":
    nb, nproc = cli_args()
    nex = int(sys.argv[3]) if len(sys.argv) > 3 else nb
    P = load_products(DATA)
    jobs = [(bi, tr, te, bi < nex) for bi, (tr, te) in enumerate(get_splits(P, nb))]
    with Pool(nproc) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    agg = {}
    keys = set().union(*[r.keys() for r in allres])
    for key in keys:
        rows = [r[key] for r in allres if key in r and r[key] is not None]
        agg[key] = mean_rows(rows)
    save("matching.json", agg, meta("run_matching", nb, n_hashes=N_HASH, grid=GRID, n_boot_exhaustive=nex))
    for key in sorted(agg, key=lambda s: (s.split("|")[0], agg[s]["frac"])):
        a = agg[key]
        print("%-24s frac=%.5f PC=%.3f PQ=%.3f | F1=%.3f P=%.3f R=%.3f | clean F1=%.3f  (n=%d)" % (
            key, a["frac"], a["PC"], a["PQ"], a["F1"], a["precision"], a["recall"], a["F1_clean"], a["n_boot"]))
