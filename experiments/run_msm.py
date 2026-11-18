"""Experiment 3: MSM, MSMP and MSMP+ re-implemented as end-to-end baselines (lshdd/msm.py).

Bootstraps over the duplicate clusters (common.get_splits, SPLIT=cluster by default). For every
bootstrap the MSM parameters (gamma, alpha, beta, mu, epsilon) are chosen by grid search on the
training products, separately for every pre-selection, and F1 is measured on the out-of-bag
products.
    msm|all          MSM on every pair of different Web shops
    msmp|r<r>        MSM on the candidates of MSMP (title model words, classical banding)
    msmp_plus|r<r>   MSM on the candidates of MSMP+ (cleaned title + value model words)
As in MSM, the similarity components of every pair are computed once for the whole data set and
shared by all bootstraps (they only depend on the two products).

Grid of the MSM paper: alpha, gamma, mu and epsilon from 0.1 to 0.9 in steps of 0.1; beta also 0,
the optimum MSM reports. kappa = lambda = 1 (key and value similarities enter linearly), as in all
run configurations of the authors' code. Epsilon stays below 1: in the authors' code pairs that are
not scored sit at distance exactly 1 and would all merge at epsilon = 1.

MSM sees the raw data (MSM_DATA=raw); the cleaning of MSMP+ only enters its LSH representation, as
in the paper. MSM_DATA=clean (MSM on cleaned data) and tune (cleaning tuned as a parameter) are
ablations.

Bootstraps: 50 for msm (MSM paper, APFA) and 100 for msmp / msmp_plus (MSMP), unless
n_bootstraps is given. Bootstrap i is the same draw in every run (one seeded sequence), so the
first 50 bootstraps of msmp are those of msm.

MSM_MODE=code (default) follows the MSM authors' own code, MSM_MODE=paper the text of the papers;
see lshdd/msm.py (MSMConfig) for the differences.

SELECTION=train picks the configuration with the best training F1. SELECTION=sparker reproduces
the MSM authors' code (core/SparkWorker.scala and core/SparkResultsAggregator.scala of
github.com/nickvalstar/APFAProject): the grid, in grid order, is cut into SPARKER_CHUNKS chunks of
SPARKER_SLICES contiguous slices; every slice picks its best configuration on training F1, and the
slices are then merged by TEST F1. That leaks the test set into the selection, so it is only
used to show how optimistic the published values are.

New variants are merged into an existing results/msm.json only when it was made with the same
protocol, grid and code; otherwise the run stops before computing anything.

usage: [SPLIT=..] [SELECTION=..] [MSM_MODE=..] [MSM_DATA=..] python experiments/run_msm.py
           [n_bootstraps, or - for the defaults] [n_processes] [variants, e.g. msm or msm,msmp]
"""
import sys, os, json, time
from dataclasses import asdict
from itertools import product as grid
from multiprocessing import Pool
from common import *
from run_matching import evaluate
from lshdd import baselines
from lshdd.msm import MSM, MODES, hsim, cluster, f1_path, same_shop

MODE = os.environ.get("MSM_MODE", "code")
if MODE not in MODES:
    raise SystemExit("MSM_MODE=%s is not one of %s" % (MODE, ", ".join(MODES)))
DATA_MODES = {"raw": (False,), "clean": (True,), "tune": (False, True)}
MSM_DATA = os.environ.get("MSM_DATA", "raw")
if MSM_DATA not in DATA_MODES:
    raise SystemExit("MSM_DATA=%s is not one of %s" % (MSM_DATA, ", ".join(DATA_MODES)))

TENTHS = tuple(round(0.1 * i, 1) for i in range(1, 10))
GAMMA = ALPHA = MU = TENTHS
BETA = (0.0,) + TENTHS
EPS = list(TENTHS)
CLEAN = DATA_MODES[MSM_DATA]
ROWS = {"msmp": (3, 4, 5, 6, 8), "msmp_plus": (3, 4, 5, 6)}
VARIANTS = ("msm", "msmp", "msmp_plus")
N_BOOT = {"msm": 50, "msmp": 100, "msmp_plus": 100}
SPARKER_CHUNKS = 10     # numChunksParams default of sparker's Runner
SPARKER_SLICES = 8      # numSlicesParams default = Spark parallelism (machine dependent)
# Results of two runs may only be merged when all of these agree.
MERGE_KEYS = ("split", "split_seed", "selection", "mode", "grid", "sparker_partitions", "code_hash")

_G = {}


def global_components(P, cfg):
    """Pairs MSM scores on the whole data set and their components (per data version in CLEAN)."""
    prods = [public(p) for p in P]
    comps = {}
    for clean in CLEAN:
        m = MSM(prods, clean=clean, cfg=cfg)
        valid = m.valid_pairs()
        comps[clean] = m.components(valid, GAMMA)
    return np.array(valid, dtype=np.int64), comps


def _init(valid, comps, sel, cfg):
    _G.update(valid=valid, comps=comps, selection=sel, cfg=cfg)


def infinite(P, ids):
    """Which pairs of a collection block merges: CODE only the pairs of one shop; PAPER (None)
    every pair that is not scored, i.e. also different brands and pairs LSH does not propose."""
    return same_shop([P[i]["shop"] for i in ids]) if _G["cfg"].unscored == "one" else None


def collection(P, ids, seed, variants):
    """Pre-selections of one collection and, per pre-selection, the indices of its pairs in the
    global components with the pairs in local positions."""
    prods = [public(P[i]) for i in ids]
    shop = [p["shop"] for p in prods]
    pre = {}
    if "msm" in variants:
        pre["msm|all"] = None
    for name, tok in (("msmp", baselines.msmp_tokens), ("msmp_plus", baselines.msmp_plus_tokenizer(prods))):
        if name in variants:
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


def settings():
    """MSM settings (clean, gamma index, alpha, beta, mu) in grid order; epsilon varies fastest
    within a setting, so configuration c is (settings()[c // len(EPS)], EPS[c % len(EPS)])."""
    return list(grid(CLEAN, range(len(GAMMA)), ALPHA, BETA, MU))


def sparker_choice(f_train, f_test):
    """Configuration chosen by sparker: best training F1 within every contiguous slice of the grid
    (first maximum, as findBestAlgorithm), then the slice winner with the best test F1."""
    winners = []
    for chunk in np.array_split(np.arange(len(f_train)), SPARKER_CHUNKS):
        for part in np.array_split(chunk, SPARKER_SLICES):
            if len(part):
                winners.append(int(part[np.argmax(f_train[part])]))
    return max(winners, key=f_test)


def one_bootstrap(args):
    bi, train, test, variants = args
    P = load_products(DATA)
    comps, cfg = _G["comps"], _G["cfg"]
    g_tr, g_te = local_gold(P, train), local_gold(P, test)
    gc_te = cleaned_gold(P, test)
    inf_tr, inf_te = infinite(P, train), infinite(P, test)
    _, sel_tr = collection(P, train, 500 + bi, variants)
    pre_te, sel_te = collection(P, test, 900 + bi, variants)
    S = settings()
    f_tr = {v: np.empty((len(S), len(EPS))) for v in sel_tr}
    for s, (clean, g, a, b, mu) in enumerate(S):
        d = 1.0 - hsim(comps[clean], g, a, b, mu, cfg)
        for v, (idx, pairs) in sel_tr.items():
            f_tr[v][s] = f1_path(len(train), pairs, d[idx], g_tr, EPS, inf_tr)
    res = {}
    for v in sel_tr:
        flat = f_tr[v].ravel()
        if _G["selection"] == "train":
            c = int(np.argmax(flat))
        else:
            idx, pairs = sel_te[v]
            test_f1 = {}

            def f_test(c):
                s = c // len(EPS)
                if s not in test_f1:
                    clean, g, a, b, mu = S[s]
                    d = 1.0 - hsim(comps[clean], g, a, b, mu, cfg)
                    test_f1[s] = f1_path(len(test), pairs, d[idx], g_te, EPS, inf_te)
                return test_f1[s][c % len(EPS)]
            c = sparker_choice(flat, f_test)
        (clean, g, a, b, mu), eps = S[c // len(EPS)], EPS[c % len(EPS)]
        t0 = time.perf_counter()
        idx, pairs = sel_te[v]
        d = 1.0 - hsim(comps[clean], g, a, b, mu, cfg)
        pred = cluster(len(test), pairs, d[idx], eps, inf_te)
        row = evaluate(pred, pre_te[v], g_te, gc_te, len(test))
        row.update(F1_train=flat[c], clean=float(clean), gamma=GAMMA[g], alpha=a, beta=b, mu=mu, epsilon=eps,
                   cluster_seconds=time.perf_counter() - t0)
        res[v] = row
    return res


def run_meta(n_boot, sel):
    return meta("run_msm", max(n_boot.values()), selection=sel, mode=MODE, msm=asdict(MODES[MODE]),
                n_boot_by_variant=n_boot,
                grid=dict(data=MSM_DATA, clean=CLEAN, gamma=GAMMA, alpha=ALPHA, beta=BETA, mu=MU, epsilon=EPS, rows=ROWS),
                sparker_partitions=[SPARKER_CHUNKS, SPARKER_SLICES] if sel == "sparker" else None)


def mergeable(path, m, variants):
    """Results (and bootstrap counts) of earlier runs to keep: those of other variants, if made
    under the same settings. Stops the run when they were made under other settings (before any
    work is done)."""
    if not os.path.exists(path) or set(variants) == set(VARIANTS):
        return {}, {}
    old, om = load_results(os.path.basename(path))
    m = json.loads(json.dumps(m))                     # tuples -> lists, as read back from JSON
    diff = [k for k in MERGE_KEYS if om is None or om.get(k) != m.get(k)]
    if diff:
        raise SystemExit("%s was made with different settings (%s); re-run all variants "
                         "(variants msm,msmp,msmp_plus) or move the file away." % (path, ", ".join(diff)))
    kept = {k: v for k, v in old.items() if k.split("|")[0] not in variants}
    return kept, {v: n for v, n in om["n_boot_by_variant"].items() if v not in variants}


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    variants = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else VARIANTS
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit("unknown variants: %s" % ", ".join(sorted(unknown)))
    sel = selection(SELECTIONS)
    P = load_products(DATA)
    n_boot = {v: nb or N_BOOT[v] for v in variants}
    m = run_meta(n_boot, sel)
    keep, kept_n = mergeable(os.path.join(RESULTS, "msm.json"), m, variants)
    cfg = MODES[MODE]
    valid, comps = global_components(P, cfg)
    jobs = [(bi, tr, te, tuple(v for v in variants if bi < n_boot[v]))
            for bi, (tr, te) in enumerate(get_splits(P, max(n_boot.values())))]
    with Pool(nproc, initializer=_init, initargs=(valid, comps, sel, cfg)) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    keys = dict.fromkeys(k for r in allres for k in r)
    agg = {key: mean_rows([r[key] for r in allres if key in r]) for key in keys}
    agg = {**keep, **agg}
    m["n_boot_by_variant"] = {**kept_n, **n_boot}
    m["n_boot"] = max(m["n_boot_by_variant"].values())
    save("msm.json", agg, m)
    for key in sorted(agg, key=lambda s: -agg[s]["frac"]):
        a = agg[key]
        print("%-18s frac=%.5f PC=%.3f | F1=%.3f (sd %.3f) P=%.3f R=%.3f | train F1=%.3f | clean=%.2f "
              "gamma=%.2f alpha=%.2f beta=%.2f mu=%.2f eps=%.2f (n=%d)" % (
                  key, a["frac"], a["PC"], a["F1"], a["F1_sd"], a["precision"], a["recall"], a["F1_train"],
                  a["clean"], a["gamma"], a["alpha"], a["beta"], a["mu"], a["epsilon"], a["n_boot"]))
