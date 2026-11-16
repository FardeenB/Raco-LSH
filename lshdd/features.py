"""Cheap pair features for the matching step (only computed for LSH candidate pairs)."""
import re
import numpy as np

CLASSES = ("tw", "tq", "id", "ip", "vq")
_UNIT_RE = re.compile(r'^(\d+(?:\.\d+)?)([a-z]+)$')

FEATURE_NAMES = (
    ["wj_all"]
    + ["wj_" + c for c in ("tw", "tq", "id", "vq")]
    + ["both_" + c for c in ("id", "vq")]
    + ["id_exact", "id_prefix", "id_lcp_short", "id_lcp_long", "id_conflict",
       "unit_agree", "unit_conflict", "inch_conflict", "vq_shared", "vq_min"]
)


class PairFeaturizer:
    """Holds the per-product views needed to featurise pairs of one product collection."""

    def __init__(self, products, token_sets, vocab, weights):
        self.products = products
        self.vocab = vocab
        self.w = weights
        self.idf = vocab.idf()
        cls = vocab.classes()
        self.cls_masks = {c: (cls == c) for c in CLASSES}
        self.ids = vocab.ids
        self.id_tokens = [sorted(t for c, t in s if c == "id") for s in token_sets]
        self.units = []
        for s in token_sets:
            u = {}
            for c, t in s:
                if c == "tq":
                    m = _UNIT_RE.match(t)
                    if m:
                        u.setdefault(m.group(2), set()).add(m.group(1))
            self.units.append(u)
        self.cls_sum = {c: np.array([self.w[a[self.cls_masks[c][a]]].sum() for a in self.ids])
                        for c in CLASSES}
        self.tot = np.array([self.w[a].sum() for a in self.ids])

    @staticmethod
    def _lcp(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def pair(self, i, j):
        a, b = self.ids[i], self.ids[j]
        inter = np.intersect1d(a, b, assume_unique=True)
        wi = self.w[inter]
        f = []
        tot_i = wi.sum()
        union = self.tot[i] + self.tot[j] - tot_i
        f.append(tot_i / union if union > 0 else 0.0)
        for c in ("tw", "tq", "id", "vq"):
            if c == "id":
                m = self.cls_masks["id"][inter] | self.cls_masks["ip"][inter]
                si = self.cls_sum["id"][i] + self.cls_sum["ip"][i]
                sj = self.cls_sum["id"][j] + self.cls_sum["ip"][j]
            else:
                m = self.cls_masks[c][inter]
                si, sj = self.cls_sum[c][i], self.cls_sum[c][j]
            x = wi[m].sum()
            u = si + sj - x
            f.append(x / u if u > 0 else 0.0)
        ia, ib = self.id_tokens[i], self.id_tokens[j]
        f.append(float(bool(ia) and bool(ib)))
        f.append(float(self.cls_sum["vq"][i] > 0 and self.cls_sum["vq"][j] > 0))
        exact = prefix = 0.0
        best_s = best_l = 0.0
        for x in ia:
            for y in ib:
                l = self._lcp(x, y)
                lo, hi = min(len(x), len(y)), max(len(x), len(y))
                if x == y:
                    exact = 1.0
                if l == lo and lo >= 5:
                    prefix = 1.0
                best_s = max(best_s, l / lo)
                best_l = max(best_l, l / hi)
        f += [exact, prefix, best_s, best_l,
              float(bool(ia) and bool(ib) and exact == 0 and prefix == 0)]
        ua, ub = self.units[i], self.units[j]
        agree = conflict = 0
        for u in ua.keys() & ub.keys():
            if ua[u] & ub[u]:
                agree += 1
            else:
                conflict += 1
        inch_conf = float("inch" in ua and "inch" in ub and not (ua["inch"] & ub["inch"]))
        f += [agree, conflict, inch_conf]
        mvq = self.cls_masks["vq"][inter]
        nvi = int(self.cls_masks["vq"][a].sum()); nvj = int(self.cls_masks["vq"][b].sum())
        f += [float(mvq.sum()), float(min(nvi, nvj))]
        return f

    def matrix(self, pairs):
        if not pairs:
            return np.zeros((0, len(FEATURE_NAMES)))
        return np.array([self.pair(i, j) for i, j in pairs], dtype=float)
