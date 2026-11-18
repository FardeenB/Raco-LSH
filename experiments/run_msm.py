"""Experiment 3: MSM, MSMP and MSMP+ re-implemented as end-to-end baselines (lshdd/msm.py).

Protocol of MSM (van Bezu et al., SAC 2015): bootstraps over the duplicate groups, so that the
ratio of duplicates versus non-duplicates is maintained (common.group_splits). For every bootstrap
the MSM parameters (gamma, alpha, beta, mu, epsilon, and whether MSM sees the MSMP+ cleaned data)
are chosen by grid search on the training products, separately for every pre-selection, and F1 is
measured on the out-of-bag products.
    msm|all          MSM on every pair of different Web shops
    msmp|r<r>        MSM on the candidates of MSMP (title model words, classical banding)
    msmp_plus|r<r>   MSM on the candidates of MSMP+ (cleaned title + value model words)
As in MSM, the similarity components of every pair are computed once for the whole data set and
shared by all bootstraps (they only depend on the two products).

usage: python experiments/run_msm.py [n_bootstraps] [n_processes] [variants, e.g. msm or msm,msmp]
"""
import sys, os, json, time
from itertools import product as grid
from multiprocessing import Pool
from common import *
from run_matching import evaluate
from lshdd import baselines
from lshdd.msm import MSM, hsim, cluster, f1_path

GAMMA = (0.6, 0.7, 0.8, 0.9)
ALPHA = (0.5, 0.6, 0.7, 0.8, 0.9)
BETA = (0.0, 0.2, 0.4, 0.6)
MU = (0.3, 0.5, 0.65, 0.8, 0.9)
EPS = [round(float(e), 2) for e in np.arange(0.1, 0.86, 0.05)]
ROWS = {"msmp": (3, 4, 5, 6, 8), "msmp_plus": (3, 4, 5, 6)}
VARIANTS = ("msm", "msmp", "msmp_plus")

_G = {}


def global_components(P):
    """Pairs MSM scores on the whole data set and their components (raw and cleaned)."""
    prods = [public(p) for p in P]
    comps = {}
    for clean in (False, True):
        m = MSM(prods, clean=clean)
        valid = m.valid_pairs()
        comps[clean] = m.components(valid, GAMMA)
    return np.array(valid, dtype=np.int64), comps


def _init(valid, comps, variants):
    _G.update(valid=valid, comps=comps, variants=variants)


def collection(P, ids, seed):
    """Pre-selections of one collection and, per pre-selection, the indices of its pairs in the
    global components with the pairs in local positions."""
    prods = [public(P[i]) for i in ids]
    shop = [p["shop"] for p in prods]
    pre = {}
    if "msm" in _G["variants"]:
        pre["msm|all"] = None
    for name, tok in (("msmp", baselines.msmp_tokens), ("msmp_plus", baselines.msmp_plus_tokenizer(prods))):
        if name in _G["variants"]:
            for r, c in baselines.classical_candidates(prods, tok, ROWS[name], seed).items():
                pre["%s|r%d" % (name, r)] = c
    V = _G["valid"]
    pos = -np.ones(len(P), dtype=np.int64)
    pos[ids] = np.arange(len(ids))
    keep = np.nonzero((pos[V[:, 0]] >= 0) & (pos[V[:, 1]] >= 0))[0]
    local = [(int(pos[a]), int(pos[b])) for a, b in V[keep]]
    sel = {}
    for v, c in pre.items():
        if c is None:
            pre[v] = {(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids)) if shop[i] != shop[j]}
            sel[v] = (keep, local)
        else:
            k = [t for t, p in enumerate(local) if p in c]
            sel[v] = (keep[k], [local[t] for t in k])
    return pre, sel


def one_bootstrap(args):
    bi, train, test = args
    P = load_products(DATA)
    comps = _G["comps"]
    g_tr, g_te = local_gold(P, train), local_gold(P, test)
    gc_te = cleaned_gold(P, test)
    _, sel_tr = collection(P, train, 500 + bi)
    pre_te, sel_te = collection(P, test, 900 + bi)
    best = {v: (-1.0, None) for v in sel_tr}
    for clean in (False, True):
        for g, a, b, mu in grid(range(len(GAMMA)), ALPHA, BETA, MU):
            d = 1.0 - hsim(comps[clean], g, a, b, mu)
            for v, (idx, pairs) in sel_tr.items():
                f = f1_path(len(train), pairs, d[idx], g_tr, EPS)
                k = int(np.argmax(f))
                if f[k] > best[v][0]:
                    best[v] = (f[k], (clean, g, a, b, mu, EPS[k]))
    res = {}
    for v, (f_tr, (clean, g, a, b, mu, eps)) in best.items():
        t0 = time.perf_counter()
        idx, pairs = sel_te[v]
        d = 1.0 - hsim(comps[clean], g, a, b, mu)
        pred = cluster(len(test), pairs, d[idx], eps)
        row = evaluate(pred, pre_te[v], g_te, gc_te, len(test))
        row.update(F1_train=f_tr, clean=float(clean), gamma=GAMMA[g], alpha=a, beta=b, mu=mu, epsilon=eps,
                   cluster_seconds=time.perf_counter() - t0)
        res[v] = row
    return res


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    variants = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else VARIANTS
    P = load_products(DATA)
    valid, comps = global_components(P)
    jobs = [(bi, tr, te) for bi, (tr, te) in enumerate(group_splits(P, nb))]
    with Pool(nproc, initializer=_init, initargs=(valid, comps, variants)) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    agg = {key: mean_rows([r[key] for r in allres]) for key in allres[0]}
    path = os.path.join(RESULTS, "msm.json")
    if os.path.exists(path):
        with open(path) as f:
            agg = {**{k: v for k, v in json.load(f).items() if k.split("|")[0] not in variants}, **agg}
    save("msm.json", agg)
    for key in sorted(agg, key=lambda s: -agg[s]["frac"]):
        a = agg[key]
        print("%-18s frac=%.5f PC=%.3f | F1=%.3f (sd %.3f) P=%.3f R=%.3f | train F1=%.3f | clean=%.2f "
              "gamma=%.2f alpha=%.2f beta=%.2f mu=%.2f eps=%.2f (n=%d)" % (
                  key, a["frac"], a["PC"], a["F1"], a["F1_sd"], a["precision"], a["recall"], a["F1_train"],
                  a["clean"], a["gamma"], a["alpha"], a["beta"], a["mu"], a["epsilon"], a["n_boot"]))
