"""Re-implementation of the Multi-component Similarity Method (MSM; van Bezu et al., SAC 2015), the
matcher of MSMP (van Dam et al., SAC 2016) and MSMP+ (Hartveld et al., CAiSE 2018).

For two products of different Web shops and not of different brands (Algorithm 1 of MSM):
    avgSim    q-gram (q=3) similarity of the values of matching keys (key similarity > gamma),
              weighted by the key similarity; m = number of key matches
    mwPerc    share of matching model words in the values whose key matched nothing
    titleSim  similarity of the Title Model Words Method (TMWM), -1 when TMWM rejects the titles
    titleSim = -1:  hSim = th1 avgSim + (1 - th1) mwPerc,           th1 = m / minFeatures
    otherwise:      hSim = th1 avgSim + th2 mwPerc + mu titleSim,   th1 = (1 - mu) m / minFeatures,
                                                                     th2 = 1 - mu - th1
All other pairs (and, for MSMP / MSMP+, pairs not proposed by LSH) are at distance infinity.
Adapted single linkage merges the two closest clusters until they are further apart than epsilon;
clusters are never merged when any pair across them is at distance infinity.

TMWM (Vandic et al., DSS 2012; as summarised by de Bakker et al., CAiSE 2013):
    title cosine > alpha                                    -> 1
    two title model words with approximately equal non-numeric parts (normalised Levenshtein
    distance < 0.5) but different numeric parts             -> -1
    otherwise  s = W_COS cosine + (1 - W_COS) avgLvSim(title words); when model words with equal
    numeric parts exist, s = DELTA avgLvSim(those model words) + (1 - DELTA) s;  s if s > beta else -1

Choices where the papers are not specific: W_COS = DELTA = 0.5; th1 = m / minFeatures is capped
at 1 (one key can match several keys); mwPerc = |C & D| / |C | D|; value model words are tokens
that contain a digit.

The TMWM conflict rule (model words with approximately equal non-numeric but different numeric
parts -> -1) is not applied by default (CONFLICT = "ignore"). MSM reports that with the optimal
beta = 0 "titleSim is never set to -1", which can only hold without the rule: on this data it
fires for 75% of the cross-shop pairs and for 33% of the duplicates, mostly on the two screen
sizes of a title ('32" Class 3112" Diag.'). "drop" (-1, the literal reading) and "zero" (title
similarity 0) are kept for ablations.
"""
import math
import re
from collections import Counter
import numpy as np

from .baselines import _MW_TITLE, _clean
from .brands import brand_of

Q = 3
W_COS = 0.5
DELTA = 0.5
APPROX = 0.5
CONFLICT = "ignore"
_STRIP = re.compile(r"[\s,/]+")
_WORDS = re.compile(r"[\s,/|()\[\]]+")


def qgrams(s):
    s = "#" * (Q - 1) + s + "#" * (Q - 1)
    return Counter(s[i:i + Q] for i in range(len(s) - Q + 1))


def qgram_sim(a, b):
    """(n1 + n2 - qGramDistance) / (n1 + n2) for two q-gram multisets."""
    n = sum(a.values()) + sum(b.values())
    if n == 0:
        return 0.0
    return (n - sum(abs(a[g] - b[g]) for g in a.keys() | b.keys())) / n


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _value_mw(v):
    out = set()
    for t in v.split():
        t = t.strip(".,;:()[]\"'!?")
        if any(c.isdigit() for c in t):
            out.add(t)
    return out


class MSM:
    """Per-pair similarity components of one product collection. The components do not depend
    on (alpha, beta, mu), so the parameter grid only costs one vectorised hsim() per setting."""

    def __init__(self, products, clean=False):
        self.n = len(products)
        self.shop = [p["shop"] for p in products]
        self.brand = [brand_of(p) for p in products]
        norm = _clean if clean else str.lower
        key_ids, val_ids = {}, {}
        self.kvp = []
        for p in products:
            ks, vs, mws = [], [], []
            for k, v in p["features"].items():
                v = norm(v)
                ks.append(key_ids.setdefault(_STRIP.sub("", norm(k)), len(key_ids)))
                vs.append(val_ids.setdefault(_STRIP.sub("", v), len(val_ids)))
                mws.append(frozenset(_value_mw(v)))
            self.kvp.append((np.array(ks, dtype=np.int64), vs, mws))
        kq = [qgrams(k) for k in sorted(key_ids, key=key_ids.get)]
        self.K = np.array([[qgram_sim(a, b) for b in kq] for a in kq]).reshape(len(kq), len(kq))
        self.values = sorted(val_ids, key=val_ids.get)
        self._vq, self._vsim, self._lv = {}, {}, {}
        self.words, self.mw = [], []
        for p in products:
            t = norm(p["title"])
            self.words.append({w for w in _WORDS.split(t) if any(c.isalnum() for c in w)})
            self.mw.append([(w, re.sub(r"[0-9]", "", w), re.sub(r"[^0-9]", "", w))
                            for w in set(_MW_TITLE.findall(t))])

    def valid_pairs(self):
        """Pairs (i, j), i<j, that MSM scores: different shops and not of different brands."""
        out = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.shop[i] != self.shop[j] and not (
                        self.brand[i] and self.brand[j] and self.brand[i] != self.brand[j]):
                    out.append((i, j))
        return out

    def _value_sim(self, a, b):
        key = (a, b) if a < b else (b, a)
        s = self._vsim.get(key)
        if s is None:
            qa = self._vq.get(a) or self._vq.setdefault(a, qgrams(self.values[a]))
            qb = self._vq.get(b) or self._vq.setdefault(b, qgrams(self.values[b]))
            s = self._vsim[key] = qgram_sim(qa, qb)
        return s

    def _lv_dist(self, a, b):
        key = (a, b) if a < b else (b, a)
        d = self._lv.get(key)
        if d is None:
            m = max(len(a), len(b))
            d = self._lv[key] = levenshtein(a, b) / m if m else 0.0
        return d

    def _avg_lv_sim(self, pairs):
        num = den = 0.0
        for x, y in pairs:
            w = len(x) + len(y)
            num += (1.0 - self._lv_dist(x, y)) * w
            den += w
        return num / den if den else 0.0

    def components(self, pairs, gammas):
        """Arrays over `pairs`: avg[g], m[g], mwp[g] for every gamma, and the TMWM ingredients."""
        P, G = len(pairs), len(gammas)
        c = dict(avg=np.zeros((G, P)), m=np.zeros((G, P)), mwp=np.zeros((G, P)), minf=np.zeros(P),
                 cos=np.zeros(P), conflict=np.zeros(P, dtype=bool), base=np.zeros(P),
                 mwsim=np.full(P, np.nan))
        for t, (i, j) in enumerate(pairs):
            ki, vi, mi = self.kvp[i]
            kj, vj, mj = self.kvp[j]
            c["minf"][t] = min(len(ki), len(kj))
            S = self.K[np.ix_(ki, kj)] if len(ki) and len(kj) else np.zeros((len(ki), len(kj)))
            for g, gamma in enumerate(gammas):
                a, b = np.nonzero(S > gamma)
                if len(a):
                    w = S[a, b]
                    vs = np.array([self._value_sim(vi[x], vj[y]) for x, y in zip(a, b)])
                    c["avg"][g, t] = (w * vs).sum() / w.sum()
                    c["m"][g, t] = len(a)
                ma, mb = set(a.tolist()), set(b.tolist())
                C = set().union(*(mi[x] for x in range(len(ki)) if x not in ma))
                D = set().union(*(mj[y] for y in range(len(kj)) if y not in mb))
                if C or D:
                    c["mwp"][g, t] = len(C & D) / len(C | D)
            A, B = self.words[i], self.words[j]
            c["cos"][t] = len(A & B) / math.sqrt(len(A) * len(B)) if A and B else 0.0
            same = []
            for x, xn, xd in self.mw[i]:
                for y, yn, yd in self.mw[j]:
                    if self._lv_dist(xn, yn) < APPROX:
                        if xd != yd:
                            c["conflict"][t] = True
                        else:
                            same.append((x, y))
            c["base"][t] = self._avg_lv_sim([(x, y) for x in A for y in B])
            if same:
                c["mwsim"][t] = self._avg_lv_sim(same)
        return c


def title_sim(c, alpha, beta, conflict=CONFLICT):
    s = W_COS * c["cos"] + (1 - W_COS) * c["base"]
    s = np.where(np.isnan(c["mwsim"]), s, DELTA * np.nan_to_num(c["mwsim"]) + (1 - DELTA) * s)
    s = np.where(s <= beta, -1.0, s)
    if conflict == "drop":
        s = np.where(c["conflict"], -1.0, s)
    elif conflict == "zero":
        s = np.where(c["conflict"], 0.0, s)
    return np.where(c["cos"] > alpha, 1.0, s)


def hsim(c, g, alpha, beta, mu, conflict=CONFLICT):
    """hSim for every pair of the components `c`, with gamma = gammas[g]."""
    t = title_sim(c, alpha, beta, conflict)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(c["minf"] > 0, np.minimum(1.0, c["m"][g] / c["minf"]), 0.0)
    avg, mwp = c["avg"][g], c["mwp"][g]
    th1 = (1 - mu) * r
    return np.where(t == -1, r * avg + (1 - r) * mwp, th1 * avg + (1 - mu - th1) * mwp + mu * t)


def _merges(n, pairs, dist, max_eps):
    """Adapted single linkage: yields (distance, cluster A, cluster B) for every merge, in order.
    `pairs` are the only pairs at finite distance; merging stops beyond max_eps."""
    finite = set(pairs)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in np.argsort(dist, kind="stable"):
        if dist[t] > max_eps:
            break
        ri, rj = find(pairs[t][0]), find(pairs[t][1])
        if ri == rj:
            continue
        A, B = members[ri], members[rj]
        if all(((a, b) if a < b else (b, a)) in finite for a in A for b in B):
            yield dist[t], A, B
            parent[rj] = ri
            members[ri] = A + B
            del members[rj]


def cluster(n, pairs, dist, eps):
    """Pairs (i, j), i<j, that end up in the same cluster."""
    out = set()
    for _, A, B in _merges(n, pairs, dist, eps):
        out.update((a, b) if a < b else (b, a) for a in A for b in B)
    return out


def f1_path(n, pairs, dist, gold, epsilons):
    """F1 of the clustering for every threshold in the ascending list `epsilons` (one pass)."""
    out, tp, npred, k = [], 0, 0, 0
    for d, A, B in _merges(n, pairs, dist, epsilons[-1]):
        while d > epsilons[k]:
            out.append(2 * tp / (npred + len(gold)) if npred + len(gold) else 0.0)
            k += 1
        npred += len(A) * len(B)
        tp += sum(((a, b) if a < b else (b, a)) in gold for a in A for b in B)
    while len(out) < len(epsilons):
        out.append(2 * tp / (npred + len(gold)) if npred + len(gold) else 0.0)
    return out
