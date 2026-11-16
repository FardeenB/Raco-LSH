"""Prints the tables of the paper and writes paper/numbers.tex (LaTeX macros), so that every
number in the text comes straight from results/*.json.

usage: python experiments/make_tables.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.join(HERE, "..", "paper")
macros = {}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def rows(res, prefix, k1_only=False):
    out = []
    for k, v in res.items():
        if k.startswith(prefix) and v["frac"] > 0:
            if k1_only and not k.endswith("|k1"):
                continue
            out.append(dict(v, key=k))
    return out


def best_f1star(rs):
    return max(rs, key=lambda r: r["F1star"])


def cheapest(rs, pc):
    """Fraction of comparisons at which the PC curve reaches `pc`: log-linear interpolation
    between neighbouring configurations on the (fraction, PC) frontier, so that a coarse
    parameter grid does not penalise a method. Returns a dict with 'frac' (and 'PQ' of the
    first configuration that reaches the target) or None."""
    import math
    front, best = [], -1.0
    for r in sorted(rs, key=lambda r: (r["frac"], -r["PC"])):
        if r["PC"] > best + 1e-12:
            front.append(r); best = r["PC"]
    for i, r in enumerate(front):
        if r["PC"] >= pc:
            if i == 0:
                return dict(frac=r["frac"], PQ=r["PQ"])
            a = front[i - 1]
            t = (pc - a["PC"]) / (r["PC"] - a["PC"])
            f = math.exp(math.log(a["frac"]) + t * (math.log(r["frac"]) - math.log(a["frac"])))
            return dict(frac=f, PQ=r["PQ"])
    return None


def pct(x, d=2):
    return ("%." + str(d) + "f") % (100 * x)


def m(name, value):
    macros[name] = value


def blocking():
    b = load("blocking.json")
    if b is None:
        return
    print("== LSH pre-selection (test collections, %d bootstraps) ==" % b["ours|r3|k3"]["n_boot"])
    m("nboot", str(b["ours|r3|k3"]["n_boot"]))
    table = []
    variants = [
        ("MSMP", "msmp|", "msmp"), ("MSMP+", "msmp_plus|", "msmpp"), ("RaCo-LSH", "ours|r3|", "ours"),
        ("  unweighted (gamma=0)", "ours_gamma0_unweighted|r3|", "abUnw"),
        ("  gamma=1", "ours_gamma1|r3|", "abGone"), ("  gamma=3", "ours_gamma3|r3|", "abGthree"),
        ("  no prefix expansion", "ours_no_prefix|r3|", "abNoPrefix"),
        ("  no value quantities", "ours_no_quantities|r3|", "abNoQty"),
        ("  title words only", "ours_title_words_only|r3|", "abTitle"),
        ("  classical banding (k=1)", "ours|", "abBanding"),
    ]
    for label, prefix, tag in variants:
        rs = rows(b, prefix, k1_only=(tag == "abBanding"))
        bf = best_f1star(rs)
        c80, c90 = cheapest(rs, 0.80), cheapest(rs, 0.90)
        print("%-28s max F1*=%.3f (PC=%.3f PQ=%.3f at %s%%) | PC>=0.80 at %s%% | PC>=0.90 at %s%%" % (
            label, bf["F1star"], bf["PC"], bf["PQ"], pct(bf["frac"], 3),
            pct(c80["frac"], 3) if c80 else "-", pct(c90["frac"], 3) if c90 else "-"))
        m(tag + "Fstar", "%.3f" % bf["F1star"]); m(tag + "FstarPC", "%.3f" % bf["PC"])
        m(tag + "FstarPQ", "%.3f" % bf["PQ"]); m(tag + "FstarFrac", pct(bf["frac"], 3))
        m(tag + "FracEighty", pct(c80["frac"], 3 if c80["frac"] < 0.01 else 1) if c80 else "--")
        m(tag + "FracNinety", pct(c90["frac"], 2) if c90 else "--")
        table.append((tag, c80, c90))
    d = dict((t, (a, c)) for t, a, c in table)
    for tag in ("msmp", "msmpp"):
        if d[tag][0] and d["ours"][0]:
            m(tag + "ReductionEighty", "%.0f" % (round(d[tag][0]["frac"] / d["ours"][0]["frac"], -1)))
            print("comparisons needed for PC>=0.80: %s / ours = %.0fx" % (tag, d[tag][0]["frac"] / d["ours"][0]["frac"]))
    for k in (1, 2, 3, 8, 12):
        r = b["ours|r3|k%d" % k]
        m("oursK%sFrac" % {1: "one", 2: "two", 3: "three", 8: "eight", 12: "twelve"}[k], pct(r["frac"], 3))
        m("oursK%sPC" % {1: "one", 2: "two", 3: "three", 8: "eight", 12: "twelve"}[k], "%.3f" % r["PC"])
        m("oursK%sPQ" % {1: "one", 2: "two", 3: "three", 8: "eight", 12: "twelve"}[k], "%.3f" % r["PQ"])


def matching():
    mt = load("matching.json")
    if mt is None:
        return
    print("\n== Duplicate detection (F1 on out-of-bag products) ==")
    keys = sorted([k for k in mt if re.match(r"rf\|r\d+\|k\d+$", k)], key=lambda k: mt[k]["frac"])
    for k in keys + ["rf|exhaustive", "rf|selected_on_train", "lr|r3|k3", "lr|r3|k2", "wj|r3|k3", "wj|r3|k2"]:
        if k not in mt:
            continue
        a = mt[k]
        print("%-22s frac=%7s%% PC=%.3f | F1=%.3f (sd %.3f) P=%.3f R=%.3f | cleaned gold F1=%.3f P=%.3f R=%.3f | n=%d" % (
            k, pct(a["frac"], 3), a["PC"], a["F1"], a["F1_sd"], a["precision"], a["recall"],
            a["F1_clean"], a["precision_clean"], a["recall_clean"], a["n_boot"]))
    names = {"rf|r3|k3": "rfThree", "rf|r3|k2": "rfTwo", "rf|r3|k8": "rfEight", "rf|r3|k1": "rfOne",
             "rf|r3|k12": "rfTwelve", "rf|r3|k32": "rfThirtyTwo", "rf|r2|k1": "rfWide",
             "rf|exhaustive": "rfAll", "rf|selected_on_train": "rfSel",
             "lr|r3|k3": "lrThree", "wj|r3|k3": "wjThree"}
    for k, tag in names.items():
        if k not in mt:
            continue
        a = mt[k]
        m(tag + "F", "%.3f" % a["F1"]); m(tag + "P", "%.3f" % a["precision"]); m(tag + "R", "%.3f" % a["recall"])
        m(tag + "Fsd", "%.3f" % a["F1_sd"]); m(tag + "Frac", pct(a["frac"], 2 if a["frac"] > 0.01 else 3))
        m(tag + "Fclean", "%.3f" % a["F1_clean"]); m(tag + "Pclean", "%.3f" % a["precision_clean"])
        m(tag + "Rclean", "%.3f" % a["recall_clean"]); m(tag + "PC", "%.3f" % a["PC"])
        m(tag + "N", str(a["n_boot"]))
    best = max(keys, key=lambda k: mt[k]["F1"])
    print("best average F1 on the sweep:", best, "%.3f" % mt[best]["F1"])


def external():
    e = load("external.json")
    if e is None:
        return
    print("\n== LSH part on other product collections ==")
    for ds, dtag in (("Abt-Buy", "abt"), ("Amazon-Google", "amz")):
        for v, vtag in (("minhash_words", "Words"), ("typed_unweighted", "Typed"), ("typed_idf2_no_prefix", "NoPrefix"), ("ours", "Ours")):
            for banding, btag in ((True, "Band"), (False, "")):
                rs = rows(e, "%s|%s|" % (ds, v), k1_only=banding)
                if not rs:
                    continue
                bf, c90 = best_f1star(rs), cheapest(rs, 0.90)
                print("%-14s %-22s %-9s max F1*=%.3f | PC>=0.90 at %s%% (PQ=%s)" % (
                    ds, v, "banding" if banding else "counting", bf["F1star"],
                    pct(c90["frac"], 3) if c90 else "-", ("%.3f" % c90["PQ"]) if c90 else "-"))
                m(dtag + vtag + btag + "Fstar", "%.3f" % bf["F1star"])
                m(dtag + vtag + btag + "FracNinety", pct(c90["frac"], 2) if c90 else "--")


def checks():
    c = load("checks.json")
    if c is None:
        return
    print("\n== Checks and timings ==")
    for k, v in c.items():
        print("  %-32s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
    m("tPredict", "%.2f" % c["predict_seconds"]); m("tFit", "%.2f" % c["fit_seconds"])
    m("tExhaustive", "%.1f" % c["exhaustive_predict_seconds"])
    m("tFullPredict", "%.1f" % c["full_predict_seconds"]); m("tFullExhaustive", "%.0f" % c["full_exhaustive_seconds"])
    m("fullSpeedup", "%.0f" % (c["full_exhaustive_seconds"] / c["full_predict_seconds"]))
    m("altF", "%.3f" % c["F1_alternative_splits_mean"]); m("altN", "%d" % c["F1_alternative_splits_n"])


if __name__ == "__main__":
    blocking(); matching(); external(); checks()
    with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
        f.write("% generated by experiments/make_tables.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote paper/numbers.tex with %d macros" % len(macros))
