"""Experiment 4: does the LSH part transfer to other product collections?

Blocking only (no training data involved), on two public clean-clean ER benchmarks:
Abt-Buy (consumer electronics) and Amazon-Google (software), as distributed with pyJedAI
(https://github.com/AI-team-UoA/pyJedAI, data/ccer/D2 and D3).

usage: python experiments/run_external.py
"""
import csv, sys
from common import *
from lshdd.pipeline import Blocker
from lshdd.tokens import title_tokens, product_tokens

csv.field_size_limit(10 ** 9)
EXT = os.path.join(os.path.dirname(DATA), "external")
SETS = {
    "Abt-Buy": ("D2_abt.csv", "D2_buy.csv", "D2_gt.csv", "|", ("name",), ("description",)),
    "Amazon-Google": ("D3_amazon.csv", "D3_gp.csv", "D3_gt.csv", "#", ("title", "manufacturer"), ("description",)),
}
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def read(path, sep):
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter=sep, quoting=csv.QUOTE_NONE))


def load(name):
    fa, fb, fg, sep, title_cols, value_cols = SETS[name]
    A, B = read(os.path.join(EXT, fa), sep), read(os.path.join(EXT, fb), sep)
    prods = []
    for shop, rows in (("A", A), ("B", B)):
        for r in rows:
            prods.append(dict(shop=shop, title=" ".join(r.get(c) or "" for c in title_cols),
                              features={c: (r.get(c) or "") for c in value_cols}))
    gold = set()
    for g in read(os.path.join(EXT, fg), sep):
        gold.add((int(g["D1"]), len(A) + int(g["D2"])))
    return prods, gold


def tok_full(p):
    """Typed tokens of the name plus quantities of the description (our representation)."""
    return product_tokens(p)


def tok_words(p):
    """Plain word tokens of the name (what ordinary MinHash blocking would use)."""
    return {("w", t) for c, t in title_tokens(p["title"], prefixes=False)}


def tok_no_prefix(p):
    return {t for t in tok_full(p) if t[0] != "ip"}


VARIANTS = {
    "minhash_words": dict(weighted=False, tokenizer=tok_words),
    "typed_unweighted": dict(weighted=False, tokenizer=tok_full),
    "typed_idf2_no_prefix": dict(gamma=2.0, tokenizer=tok_no_prefix),
    "ours": dict(gamma=2.0, tokenizer=tok_full),
}

if __name__ == "__main__":
    out = {}
    for name in SETS:
        prods, gold = load(name)
        n = len(prods)
        na = sum(p["shop"] == "A" for p in prods)
        total = na * (n - na)                      # clean-clean ER: only cross-source pairs exist
        print(name, "products", n, "matches", len(gold), "possible comparisons", total)
        for vname, kw in VARIANTS.items():
            for r in (2, 3, 4, 6):
                rows = {}
                for seed in range(5):
                    blk = Blocker(n_hashes=1200, rows=r, seed=seed, **kw)
                    idx = blk.index(prods)
                    pairs = list(idx["counts"].keys()); cnt = np.array(list(idx["counts"].values()))
                    for k in KS:
                        c = {p for p, v in zip(pairs, cnt) if v >= k}
                        s = blocking_scores(c, gold, n)
                        s["frac"] = len(c) / total
                        rows.setdefault(k, []).append(s)
                for k, v in rows.items():
                    out["%s|%s|r%d|k%d" % (name, vname, r, k)] = mean_rows(v)
        for vname in VARIANTS:
            best = max((v for kk, v in out.items() if kk.startswith(name + "|" + vname + "|")), key=lambda a: a["F1star"])
            pcs = {}
            for target in (0.8, 0.9):
                ok = [v for kk, v in out.items() if kk.startswith(name + "|" + vname + "|") and v["PC"] >= target]
                pcs[target] = min(ok, key=lambda a: a["frac"]) if ok else None
            print("  %-24s best F1*=%.3f (PC=%.3f PQ=%.3f frac=%.5f) | " % (vname, best["F1star"], best["PC"], best["PQ"], best["frac"])
                  + "  ".join("PC>=%.1f: frac=%s PQ=%s" % (t, ("%.5f" % a["frac"]) if a else "-", ("%.3f" % a["PQ"]) if a else "-") for t, a in pcs.items()))
    save("external.json", out)
