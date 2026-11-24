"""Figures for the paper, from results/*.json.  usage: python experiments/make_figures.py"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
OUT = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        res = json.load(f)
    res.pop("_meta", None)
    return res


def series(res, prefix, pattern=None):
    rows = [v for k, v in res.items() if k.startswith(prefix) and (pattern is None or re.search(pattern, k))]
    rows = [r for r in rows if r["frac"] > 0]
    return sorted(rows, key=lambda r: r["frac"])


def frontier(rows):
    """Configurations that are not dominated in (fraction of comparisons, pair completeness)."""
    out, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r["frac"], -r["PC"])):
        if r["PC"] > best + 1e-9:
            out.append(r); best = r["PC"]
    return out


def combined_figure():
    """Panels a-c: pre-selection quality; panel d: end-to-end F1. All against the fraction of comparisons."""
    b, m = load("blocking.json"), load("matching.json")
    methods = [
        ("MSMP", frontier(series(b, "msmp|")), MUTED, "s", ":"),
        ("MSMP+", frontier(series(b, "msmp_plus|")), ORANGE, "^", "--"),
        ("RaCo-LSH", frontier(series(b, "ours|")), BLUE, "o", "-"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.8), constrained_layout=True)
    panels = [("PC", "(a) Pair completeness"), ("PQ", "(b) Pair quality"), ("F1star", "(c) F1*-measure")]
    for ax, (key, label) in zip(axes, panels):
        for name, rows, color, marker, ls in methods:
            ax.plot([r["frac"] for r in rows], [r[key] for r in rows], color=color, marker=marker,
                    linestyle=ls, label=name, markeredgecolor="white", markeredgewidth=0.4,
                    markersize=2.6 if name == "RaCo-LSH" else 3.6)
        ax.set_xscale("log"); ax.set_xlim(1e-5, 1.0); ax.set_ylim(0, 1.0)
        ax.set_xticks([1e-4, 1e-2, 1]); ax.set_title(label, fontsize=8, color=INK)
    axes[1].legend(loc="upper right", fontsize=6.8, handlelength=2.4, borderaxespad=0.1)
    ax = axes[3]
    rf = sorted([v for k, v in m.items() if re.match(r"rf\|r\d+\|k\d+$", k)], key=lambda r: r["frac"])
    wj = series(m, "wj|r3|")
    if "rf|exhaustive" in m:
        rf = rf + [m["rf|exhaustive"]]
    ax.plot([r["frac"] for r in rf], [r["F1_clean"] for r in rf], color=BLUE, linestyle="--", linewidth=0.9)
    ax.plot([r["frac"] for r in rf], [r["F1"] for r in rf], color=BLUE, marker="o", markersize=2.6,
            markeredgecolor="white", markeredgewidth=0.4)
    ax.plot([r["frac"] for r in wj], [r["F1"] for r in wj], color=AQUA, marker="D", markersize=2.4, linestyle="-.",
            markeredgecolor="white", markeredgewidth=0.4)
    ax.annotate("classifier", (rf[-4]["frac"], rf[-4]["F1"]), textcoords="offset points", xytext=(0, -9), fontsize=6.3, color=INK, ha="center")
    ax.annotate("cleaned gold", (rf[-4]["frac"], rf[-4]["F1_clean"]), textcoords="offset points", xytext=(0, 3), fontsize=6.3, color=INK, ha="center")
    ax.annotate("threshold", (wj[-1]["frac"], wj[-1]["F1"]), textcoords="offset points", xytext=(4, -2), fontsize=6.3, color=INK, ha="left", va="center")
    # re-implemented baselines (same protocol as ours) as lines; published values as open markers
    ms = load("msm.json")
    for prefix, color, marker, ls in (("msmp|", MUTED, "s", ":"), ("msmp_plus|", ORANGE, "^", "--")):
        rows = series(ms, prefix)
        ax.plot([r["frac"] for r in rows], [r["F1"] for r in rows], color=color, marker=marker, linestyle=ls,
                markersize=3.2, markeredgecolor="white", markeredgewidth=0.4)
    a = ms["msm|all"]
    ax.plot([a["frac"]], [a["F1"]], color=ORANGE, marker="o", markersize=4.2, linestyle="none",
            markeredgecolor="white", markeredgewidth=0.4, clip_on=False)
    ax.annotate("MSM", (a["frac"], a["F1"]), textcoords="offset points", xytext=(0, -5), ha="center", va="top", fontsize=6.3, color=MUTED)
    for x, y, mk in ((1.0, 0.746, "P"), (1.0, 0.525, "o"), (0.05, 0.477, "s"), (0.05, 0.49, "^")):
        ax.plot([x], [y], color=MUTED, marker=mk, markersize=4.0, linestyle="none", markerfacecolor="none",
                markeredgewidth=0.7, clip_on=False)
    for name, x, y, off, va in (("APFA (publ.)", 1.0, 0.746, (-5, 0), "center"), ("MSM (publ.)", 1.0, 0.525, (-5, 3), "bottom"),
                                ("MSMP(+) (publ.)", 0.05, 0.477, (-5, -3), "top")):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off, ha="right", va=va, fontsize=6.3, color=MUTED)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1.0); ax.set_ylim(0.2, 0.9); ax.set_xticks([1e-4, 1e-2, 1])
    ax.set_title("(d) F1-measure", fontsize=8, color=INK)
    fig.supxlabel("Fraction of comparisons (log scale)", fontsize=8, color=INK)
    fig.savefig(os.path.join(OUT, "results.pdf")); fig.savefig(os.path.join(OUT, "results.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    combined_figure()
    print("figures written to", os.path.abspath(OUT))
