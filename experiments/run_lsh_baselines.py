"""Experiment 3b: the LSH pre-selection of MSMP and MSMP+ on its own, as their papers evaluate it.

For every bootstrap the LSH is built on the bootstrap sample (the drawn products, about 63% or
1,000 products: "every bootstrap contains around 1,000 products" in MSMP and MSMP+) and swept over
t = 0.05, 0.10, ..., 0.95 (run_msm.T_GRID), for the representations of MSMP+ Fig. 2-4:
    msmp               title model words (MSMP)
    msmp_clean         title model words after the data cleaning of MSMP+          ("clean")
    msmp_clean_values  ... plus title model words that re-occur in the values      ("values")
    msmp_plus          ... plus the decimal numbers of the values (MSMP+)
MSM_MODE=code (default) | paper selects the LSH exactly as in run_msm.py.

Rows (means over the bootstraps) per variant|t:
    frac, PC, PQ, F1star, n_cand   LSH candidate pairs, pairs of one shop included, over all pairs
                                   (PC and PQ as defined by MSMP+)
    frac_scored                    pairs of different shops MSM would compare
    PC_perfect                     recall after clustering with MSMP's perfect similarity (the PC
                                   curve of MSMP, Fig. 2)
    b, r, t                        bands and threshold
The printout ends with the areas under the PC, PQ and F1* curves over frac (trapezoid rule over
the mean points, on the range of frac all variants cover) and their change against msmp; MSMP+
reports +12.2%, +9.2% and +9.3% for MSMP+.

usage: [SPLIT=..] [MSM_MODE=..] python experiments/run_lsh_baselines.py [n_bootstraps] [n_processes]
"""
import sys
from multiprocessing import Pool
from common import *
import run_msm as R
from lshdd.msm import MSM, MODES

FAMILIES = ("msmp", "msmp_clean", "msmp_clean_values", "msmp_plus")
METRICS = ("PC", "PQ", "F1star")


def one_bootstrap(args):
    bi, sample = args
    P = load_products(DATA)
    gold = local_gold(P, sample)
    inf = R.infinite(P, sample)
    out = {}
    # seed 500 + bi: the LSH of run_msm.py on the same (training) products
    for v, sel in R.collection(P, sample, 500 + bi, FAMILIES, gold).items():
        row = dict(sel["stats"])
        row["PC_perfect"] = R.pc_perfect(len(sample), sel["pairs"], gold, inf)
        out[v] = row
    return out


def areas(agg):
    """(range, {family: {metric: area}}): area under metric(frac), trapezoid rule over the mean
    points, on the range of frac that every family covers (curves interpolated at its ends).
    Over each curve's own range the areas would mostly compare how far the curves reach."""
    curves = {}
    for fam in FAMILIES:
        rows = sorted((r for k, r in agg.items() if k.split("|")[0] == fam), key=lambda r: r["frac"])
        x, first = np.unique([r["frac"] for r in rows], return_index=True)
        curves[fam] = x, [rows[i] for i in first]
    lo = max(x[0] for x, _ in curves.values())
    hi = min(x[-1] for x, _ in curves.values())
    out = {}
    for fam, (x, rows) in curves.items():
        grid = np.concatenate([[lo], x[(x > lo) & (x < hi)], [hi]])
        out[fam] = {m: float(np.trapezoid(np.interp(grid, x, [r[m] for r in rows]), grid))
                    for m in METRICS + ("PC_perfect",)}
    return (float(lo), float(hi)), out


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    selection()
    P = load_products(DATA)
    cfg = MODES[R.MODE]
    valid = np.array(MSM([public(p) for p in P], cfg=cfg).valid_pairs(), dtype=np.int64)
    jobs = [(bi, tr) for bi, (tr, te) in enumerate(get_splits(P, nb))]
    with Pool(nproc, initializer=R._init, initargs=(valid, None, "train", cfg)) as pool:
        allres = pool.map(one_bootstrap, jobs, chunksize=1)
    agg = {key: mean_rows([r[key] for r in allres]) for key in allres[0]}
    save("lsh_baselines.json", agg, meta("run_lsh_baselines", nb, mode=R.MODE, t=R.T_GRID, families=FAMILIES,
                                         sample="bootstrap sample (training products)"))
    for key in sorted(agg, key=lambda k: (k.split("|")[0], -agg[k]["frac"])):
        a = agg[key]
        print("%-24s b=%3.0f r=%3.0f frac=%.5f scored=%.5f | PC=%.3f PQ=%.4f F1*=%.4f PC_perfect=%.3f" % (
            key, a["b"], a["r"], a["frac"], a["frac_scored"], a["PC"], a["PQ"], a["F1star"], a["PC_perfect"]))
    (lo, hi), A = areas(agg)
    print("\narea under the curve over frac %.4f-%.4f (covered by every family), change against msmp:" % (lo, hi))
    for fam in FAMILIES:
        print("  %-18s " % fam + "  ".join(
            "%s %.5f (%+.1f%%)" % (m, A[fam][m], 100 * (A[fam][m] / A["msmp"][m] - 1)) for m in METRICS + ("PC_perfect",)))
