"""Experiment 3: MSM, MSMP and MSMP+ re-implemented as end-to-end baselines (lshdd/msm.py,
lshdd/baselines.py), to reproduce their values for the figure.

For every bootstrap the MSM parameters (gamma, alpha, beta, mu, epsilon) are chosen by grid search
on the training products, separately for every pre-selection, and F1 is measured on the
out-of-bag products.
    msm|all          MSM on every pair of different Web shops
    msmp|t<t>        MSM after the LSH of MSMP (title model words) with threshold t
    msmp_plus|t<t>   MSM after the LSH of MSMP+ (cleaned title + value model words)
The LSH is built on the collection at hand (training or out-of-bag products), with a signature of
50% of the vector length and t = 0.05, 0.10, ..., 0.95 as in MSMP; t fixes the bands
(baselines.bands_for). As in MSM, the similarity components of every pair are computed once for
the whole data set and shared by all bootstraps (they only depend on the two products).

Everything the papers leave open follows the authors' code (see lshdd/msm.py and
lshdd/baselines.py): MinHash by random permutations, concatenated band keys, the Java model-word
regex for MSMP; MSM scores every pair of products with a common LSH neighbour, and pairs that are
not scored stay at distance 1, so only pairs of one shop block merges. MSMP+ has no public code; it
puts the MSMP+ representation into the MSMP code. MSM sees the raw data; the cleaning of MSMP+
only enters its LSH representation, as in the paper.

Every row reports, on the out-of-bag products:
    F1, precision, recall (and *_clean against the cleaned gold standard)
    frac, PC, PQ, F1star, n_cand   LSH candidate pairs, pairs of one shop included, over all
                                   n(n-1)/2 pairs: the "fraction of comparisons" of MSMP+;
                                   msm|all has frac = 1
    frac_scored                    pairs of different shops whose similarity is computed, over
                                   all pairs: the comparisons MSM makes (simsComputed)
    PC_perfect                     recall after clustering with MSMP's "perfect similarity" (1 for
                                   scored duplicates, 0 otherwise): the PC curve of MSMP
    b, r, t                        bands and threshold

Grid of the MSM paper: alpha, gamma, mu and epsilon from 0.1 to 0.9 in steps of 0.1; beta also 0,
the optimum MSM reports. kappa = lambda = 1 (key and value similarities enter linearly), as in all
run configurations of the authors' code. Epsilon stays below 1: pairs that are not scored sit at
distance exactly 1 and would all merge at epsilon = 1. The configuration with the best training F1
is applied to the out-of-bag products.

usage: python experiments/run_msm.py [n_bootstraps] [n_processes]
"""
import time
from itertools import product as grid
from multiprocessing import Pool
from common import *
from lshdd import baselines
from lshdd.msm import MSM, hsim, title_sim, cluster, f1_paths, same_shop

TENTHS = tuple(round(0.1 * i, 1) for i in range(1, 10))
GAMMA = ALPHA = MU = TENTHS
BETA = (0.0,) + TENTHS
EPS = list(TENTHS)
T_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
# LSH representation of every pre-selection, built per collection
FAMILIES = {
    "msmp": lambda prods: baselines.msmp_tokens_code,
    "msmp_plus": lambda prods: baselines.msmp_plus_tokenizer(prods),
}

_G = {}


def global_components(P):
    """Pairs MSM scores on the whole data set and their components."""
    m = MSM([public(p) for p in P])
    valid = m.valid_pairs()
    return np.array(valid, dtype=np.int64), m.components(valid, GAMMA)


def _init(valid, comps):
    _G.update(valid=valid, comps=comps)


def lsh_stats(n_cand, found, n_gold, n_all):
    pq = found / n_cand if n_cand else 0.0
    pc = found / n_gold if n_gold else 0.0
    return dict(frac=n_cand / n_all, PC=pc, PQ=pq, F1star=2 * pq * pc / (pq + pc) if pq + pc else 0.0,
                n_cand=float(n_cand))


def collection(P, ids, seed, gold):
    """Pre-selections of one collection. Per variant: the indices of its scored pairs in the global
    components, those pairs in local positions, a key shared by variants that score the same
    pairs, and the LSH statistics."""
    prods = [public(P[i]) for i in ids]
    n, n_all = len(ids), len(ids) * (len(ids) - 1) / 2
    shop = np.array([p["shop"] for p in prods])
    cross = shop[:, None] != shop[None, :]
    V = _G["valid"]
    pos = -np.ones(len(P), dtype=np.int64)
    pos[ids] = np.arange(n)
    keep = np.nonzero((pos[V[:, 0]] >= 0) & (pos[V[:, 1]] >= 0))[0]
    ia, ib = pos[V[keep, 0]], pos[V[keep, 1]]
    local = list(zip(ia.tolist(), ib.tolist()))
    g = np.array(sorted(gold), dtype=np.int64).reshape(-1, 2)
    st = lsh_stats(n_all, len(gold), len(gold), n_all)
    st.update(frac_scored=np.triu(cross, 1).sum() / n_all, b=0, r=0, t=0.0)
    out = {"msm|all": dict(idx=keep, pairs=local, key="all", stats=st)}
    for name, make in FAMILIES.items():
        for t, (b, r, N) in baselines.lsh_neighbours(prods, make(prods), T_GRID, seed).items():
            scored = baselines.two_hop(N)
            k = np.flatnonzero(scored[ia, ib])
            st = lsh_stats((N.sum() - n) / 2, int(N[g[:, 0], g[:, 1]].sum()), len(gold), n_all)
            st.update(frac_scored=np.triu(scored & cross, 1).sum() / n_all, b=b, r=r, t=t)
            out["%s|t%.2f" % (name, t)] = dict(idx=keep[k], pairs=[local[j] for j in k], key=(name, b, r), stats=st)
    return out


def pc_perfect(n, pairs, gold, infinite):
    """Recall after clustering with MSMP's "perfect similarity": distance 0 for scored duplicates,
    1 for everything else (MSMP, Section 8; core/PerfectSimilarity in the authors' code)."""
    dup = [p for p in pairs if p in gold]
    return len(cluster(n, dup, np.zeros(len(dup)), 0.5, infinite) & gold) / len(gold) if gold else 0.0


def settings():
    """MSM settings (gamma index, alpha, beta, mu) in grid order; epsilon varies fastest within a
    setting, so configuration c is (settings()[c // len(EPS)], EPS[c % len(EPS)])."""
    return list(grid(range(len(GAMMA)), ALPHA, BETA, MU))


def one_bootstrap(args):
    bi, train, test = args
    P = load_products(DATA)
    comps = _G["comps"]
    g_tr, g_te = local_gold(P, train), local_gold(P, test)
    gc_te = cleaned_gold(P, test)
    inf_tr = same_shop([P[i]["shop"] for i in train])
    inf_te = same_shop([P[i]["shop"] for i in test])
    sel_tr = collection(P, train, 500 + bi, g_tr)
    sel_te = collection(P, test, 900 + bi, g_te)
    # variants that score the same training pairs (e.g. two t with the same bands) share the search
    shared = {sel["key"]: sel for sel in sel_tr.values()}
    # the search only needs the components of the training pairs ("need", every pair msm|all
    # scores); which of them every pre-selection scores:
    keys = list(shared)
    need = np.unique(np.concatenate([sel["idx"] for sel in shared.values()]))
    sub = {k: v[..., need] for k, v in comps.items()}
    every = sel_tr["msm|all"]
    need_pairs = [every["pairs"][j] for j in np.searchsorted(every["idx"], need)]
    member = np.zeros((len(keys), len(need)), dtype=bool)
    for row, key in enumerate(keys):
        member[row, np.searchsorted(need, shared[key]["idx"])] = True
    S = settings()
    # title_sim depends on (alpha, beta) only: one per group of settings
    groups = {}
    for s, (g, a, b, mu) in enumerate(S):
        groups.setdefault((a, b), []).append((s, g, mu))
    f_tr = {key: np.empty((len(S), len(EPS))) for key in shared}
    for (a, b), members in groups.items():
        title = title_sim(sub, a, b)
        for s, g, mu in members:
            d = 1.0 - hsim(sub, g, a, b, mu, title)
            # one sort and one compiled pass for all pre-selections (= f1_path per pre-selection)
            F = f1_paths(len(train), need_pairs, d, member, g_tr, EPS, inf_tr)
            for row, key in enumerate(keys):
                f_tr[key][s] = F[row]
    res = {}
    for v, tr in sel_tr.items():
        te = sel_te[v]
        flat = f_tr[tr["key"]].ravel()
        c = int(np.argmax(flat))
        (g, a, b, mu), eps = S[c // len(EPS)], EPS[c % len(EPS)]
        t0 = time.perf_counter()
        d = 1.0 - hsim(comps, g, a, b, mu)
        pred = cluster(len(test), te["pairs"], d[te["idx"]], eps, inf_te)
        seconds = time.perf_counter() - t0
        row = dict(prf(pred, g_te))
        cl = prf(pred, gc_te)
        row.update(F1_clean=cl["F1"], precision_clean=cl["precision"], recall_clean=cl["recall"])
        row.update(te["stats"])
        row["PC_perfect"] = pc_perfect(len(test), te["pairs"], g_te, inf_te)
        row.update(F1_train=flat[c], gamma=GAMMA[g], alpha=a, beta=b, mu=mu, epsilon=eps,
                   cluster_seconds=seconds)
        res[v] = row
    return res


if __name__ == "__main__":
    nb, nproc = cli_args()
    P = load_products(DATA)
    valid, comps = global_components(P)
    jobs = [(bi, tr, te) for bi, (tr, te) in enumerate(get_splits(P, nb))]
    with Pool(nproc, initializer=_init, initargs=(valid, comps)) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    agg = {key: mean_rows([r[key] for r in allres]) for key in allres[0]}
    save("msm.json", agg, meta("run_msm", nb, grid=dict(gamma=GAMMA, alpha=ALPHA, beta=BETA, mu=MU,
                                                         epsilon=EPS, t=T_GRID)))
    for key in sorted(agg, key=lambda s: -agg[s]["frac"]):
        a = agg[key]
        print("%-18s frac=%.5f scored=%.5f PC=%.3f PC_perfect=%.3f | F1=%.3f (sd %.3f) P=%.3f R=%.3f | train F1=%.3f | "
              "b=%.0f r=%.0f gamma=%.2f alpha=%.2f beta=%.2f mu=%.2f eps=%.2f (n=%d)" % (
                  key, a["frac"], a["frac_scored"], a["PC"], a["PC_perfect"], a["F1"], a["F1_sd"], a["precision"],
                  a["recall"], a["F1_train"], a["b"], a["r"], a["gamma"], a["alpha"], a["beta"], a["mu"],
                  a["epsilon"], a["n_boot"]))
