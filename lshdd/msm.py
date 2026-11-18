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
Adapted single linkage merges the two closest clusters until they are further apart than epsilon;
clusters are never merged when any pair across them is at distance infinity.

Two presets of MSMConfig (MODES):
  PAPER  the papers' text. Pairs of the same shop, of different brands and (MSMP / MSMP+) pairs
         that LSH does not propose are at distance infinity. Choices where the text is not
         specific: W_COS = DELTA = 0.5; th1 capped at 1 (one key can match several keys);
         mwPerc = |C & D| / |C | D|; value model words are tokens that contain a digit; the TMWM
         conflict rule is not applied (CONFLICT, see below).
  CODE   sparker, the MSM authors' own code (msm/MsmSimilarity.java, msm/TitleAnalyzerImpl.java,
         msm/MsmClusteringProcedure.java of github.com/nickvalstar/APFAProject), which produced
         the MSM values reported by MSMP, MSMP+ and APFA:
           - text is lower-cased and ", ", ". ", a final "." and brackets become spaces
             (white space is kept)
           - th1 is not capped
           - mwPerc = (|C in D| + |D in C|) / (|C| + |D|) over lists that keep duplicates (Dice)
           - value model words: the title model-word regex on the raw value
           - title cosine over word counts of the title split on white space
           - TMWM conflict rule applied; the Levenshtein part averages over the title model
             words, and the model-word part over pairs with equal letters and approximately
             equal numbers; when either average is empty (0/0) TMWM returns -1
           - brands: a list of 37 brands matched against the title words; products of different
             brands get similarity 0, i.e. distance 1, not infinity
           - pairs that are not scored stay at distance 1, so only pairs of the same shop are at
             distance infinity and block merges
         Distances of exactly 1 only matter for epsilon >= 1, which the grids exclude, so pairs
         of different brands are simply left unscored.

TMWM conflict rule (model words with approximately equal non-numeric but different numeric parts
-> -1): PAPER does not apply it (CONFLICT = "ignore"), because MSM reports that with the optimal
beta = 0 "titleSim is never set to -1", which can only hold without the rule; on this data it
fires for 75% of the cross-shop pairs and for 33% of the duplicates, mostly on the two screen
sizes of a title ('32" Class 3112" Diag.'). The authors' code applies it all the same ("drop").
"zero" (title similarity 0) is kept for ablations.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import chain
import numpy as np

from .baselines import _MW_JAVA, _MW_TITLE, _clean
from .brands import brand_of, title_brands

Q = 3
W_COS = 0.5
DELTA = 0.5
APPROX = 0.5
CONFLICT = "ignore"
_STRIP = re.compile(r"[\s,/]+")
_WORDS = re.compile(r"[\s,/|()\[\]]+")
_SPARKER_PRE = re.compile(r", |\. |\.$|\(|\)")
_JAVA_WS = re.compile(r"[ \t\n\x0b\f\r]+")


@dataclass(frozen=True)
class MSMConfig:
    text: str = "strip"         # strip: lower case, no white space / commas / slashes | sparker
    theta_cap: bool = True      # cap th1 = m / minFeatures at 1
    mwperc: str = "jaccard"     # jaccard over sets | dice over lists (sparker)
    value_mw: str = "digit"     # digit: tokens with a digit | regex: title model-word regex (sparker)
    title_cos: str = "set"      # set: cosine of word sets | tf: of word counts (sparker)
    conflict: str = CONFLICT    # ignore | drop | zero, see module docstring
    tmwm: str = "paper"         # paper | sparker: Levenshtein parts as in TitleAnalyzerImpl
    brands: str = "repo"        # repo: lshdd/brands.py | sparker: 37 brands of the title words
    unscored: str = "inf"       # distance of pairs that are not scored: inf | one (sparker)


PAPER = MSMConfig()
CODE = MSMConfig(text="sparker", theta_cap=False, mwperc="dice", value_mw="regex", title_cos="tf",
                 conflict="drop", tmwm="sparker", brands="sparker", unscored="one")
MODES = {"paper": PAPER, "code": CODE}


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


def sparker_text(s):
    """preProcess() of MsmSimilarity / TitleAnalyzerImpl."""
    return _SPARKER_PRE.sub(" ", s).lower()


def java_split(s):
    """s.split("\\\\s+") in Java: trailing empty strings are dropped."""
    out = _JAVA_WS.split(s)
    while len(out) > 1 and out[-1] == "":
        out.pop()
    return out


def dice(C, D):
    """mw() of MsmSimilarity: matching model words over lists that keep duplicates."""
    if not C and not D:
        return 0.0
    sc, sd = set(C), set(D)
    return (sum(x in sd for x in C) + sum(y in sc for y in D)) / (len(C) + len(D))


def _value_mw(v):
    out = set()
    for t in v.split():
        t = t.strip(".,;:()[]\"'!?")
        if any(c.isdigit() for c in t):
            out.add(t)
    return out


def _model_words_java(s):
    return {m.group(1) for m in _MW_JAVA.finditer(s)}


class MSM:
    """Per-pair similarity components of one product collection. The components do not depend
    on (alpha, beta, mu), so the parameter grid only costs one vectorised hsim() per setting."""

    def __init__(self, products, clean=False, cfg=PAPER):
        self.cfg = cfg
        self.n = len(products)
        self.shop = [p["shop"] for p in products]
        if cfg.brands == "sparker":
            self.brands = [title_brands(p["title"]) for p in products]
        else:
            self.brands = [frozenset([b]) if b else frozenset() for b in map(brand_of, products)]
        sparker = cfg.text == "sparker"
        if sparker:
            base = _clean if clean else (lambda s: s)
            norm, key_norm, val_norm = base, (lambda k: sparker_text(base(k))), (lambda v: sparker_text(base(v)))
        else:
            norm = _clean if clean else str.lower
            key_norm, val_norm = (lambda k: _STRIP.sub("", norm(k))), (lambda v: _STRIP.sub("", norm(v)))
        value_mw = _model_words_java if cfg.value_mw == "regex" else _value_mw
        key_ids, val_ids = {}, {}
        self.kvp = []
        for p in products:
            ks, vs, mws = [], [], []
            for k, v in p["features"].items():
                ks.append(key_ids.setdefault(key_norm(k), len(key_ids)))
                vs.append(val_ids.setdefault(val_norm(v), len(val_ids)))
                # sparker extracts the value model words from the raw value
                mws.append(frozenset(value_mw(norm(v))))
            self.kvp.append((np.array(ks, dtype=np.int64), vs, mws))
        kq = [qgrams(k) for k in sorted(key_ids, key=key_ids.get)]
        self.K = np.array([[qgram_sim(a, b) for b in kq] for a in kq]).reshape(len(kq), len(kq))
        self.values = sorted(val_ids, key=val_ids.get)
        self._vq, self._vsim, self._lv = {}, {}, {}
        self.words, self.mw = [], []
        for p in products:
            if sparker:
                t = sparker_text(norm(p["title"]))
                self.words.append(Counter(java_split(t)))
                mws = _model_words_java(t)
            else:
                t = norm(p["title"])
                self.words.append({w for w in _WORDS.split(t) if any(c.isalnum() for c in w)})
                mws = set(_MW_TITLE.findall(t))
            # (word, non-numeric part, numeric part, letters)
            self.mw.append([(w, re.sub(r"[0-9]", "", w), re.sub(r"[^0-9]", "", w), re.sub(r"[^a-zA-Z]", "", w))
                            for w in mws])

    def different_brands(self, i, j):
        return bool(self.brands[i] and self.brands[j] and not (self.brands[i] & self.brands[j]))

    def valid_pairs(self):
        """Pairs (i, j), i<j, that MSM scores: different shops and not of different brands."""
        out = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.shop[i] != self.shop[j] and not self.different_brands(i, j):
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

    def _avg_lv_sim(self, pairs, empty=0.0):
        num = den = 0.0
        for x, y in pairs:
            w = len(x) + len(y)
            num += (1.0 - self._lv_dist(x, y)) * w
            den += w
        return num / den if den else empty

    def _title_cos(self, A, B):
        if self.cfg.title_cos == "tf":
            dot = sum(c * B[w] for w, c in A.items())
            na, nb = sum(c * c for c in A.values()), sum(c * c for c in B.values())
            return dot / math.sqrt(na * nb) if na and nb else 0.0
        A, B = set(A), set(B)
        return len(A & B) / math.sqrt(len(A) * len(B)) if A and B else 0.0

    def components(self, pairs, gammas):
        """Arrays over `pairs`: avg[g], m[g], mwp[g] for every gamma, and the TMWM ingredients."""
        cfg = self.cfg
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
                if cfg.mwperc == "dice":
                    C = list(chain.from_iterable(mi[x] for x in range(len(ki)) if x not in ma))
                    D = list(chain.from_iterable(mj[y] for y in range(len(kj)) if y not in mb))
                    c["mwp"][g, t] = dice(C, D)
                else:
                    C = set().union(*(mi[x] for x in range(len(ki)) if x not in ma))
                    D = set().union(*(mj[y] for y in range(len(kj)) if y not in mb))
                    if C or D:
                        c["mwp"][g, t] = len(C & D) / len(C | D)
            A, B = self.words[i], self.words[j]
            c["cos"][t] = self._title_cos(A, B)
            same = []
            for x, xn, xd, xl in self.mw[i]:
                for y, yn, yd, yl in self.mw[j]:
                    if self._lv_dist(xn, yn) < APPROX and xd != yd:
                        c["conflict"][t] = True
                    if cfg.tmwm == "sparker":
                        if xl == yl and self._lv_dist(xd, yd) < APPROX:
                            same.append((x, y))
                    elif self._lv_dist(xn, yn) < APPROX and xd == yd:
                        same.append((x, y))
            if cfg.tmwm == "sparker":
                # avgLvSim / avgLvSimMW of TitleAnalyzerImpl: 0/0 = NaN when nothing is averaged
                c["base"][t] = self._avg_lv_sim([(x, y) for x, *_ in self.mw[i] for y, *_ in self.mw[j]], np.nan)
                c["mwsim"][t] = self._avg_lv_sim(same, np.nan)
            else:
                c["base"][t] = self._avg_lv_sim([(x, y) for x in A for y in B])
                if same:
                    c["mwsim"][t] = self._avg_lv_sim(same)
        return c


def title_sim(c, alpha, beta, cfg=PAPER):
    if cfg.tmwm == "sparker":
        # titleLV: both averages always enter; NaN (empty average) fails "> beta" and gives -1
        s = 0.5 * c["mwsim"] + 0.5 * (0.5 * c["cos"] + 0.5 * c["base"])
        with np.errstate(invalid="ignore"):
            s = np.where(np.isnan(s) | (s <= beta), -1.0, s)
    else:
        s = W_COS * c["cos"] + (1 - W_COS) * c["base"]
        s = np.where(np.isnan(c["mwsim"]), s, DELTA * np.nan_to_num(c["mwsim"]) + (1 - DELTA) * s)
        s = np.where(s <= beta, -1.0, s)
    if cfg.conflict == "drop":
        s = np.where(c["conflict"], -1.0, s)
    elif cfg.conflict == "zero":
        s = np.where(c["conflict"], 0.0, s)
    return np.where(c["cos"] > alpha, 1.0, s)


def hsim(c, g, alpha, beta, mu, cfg=PAPER):
    """hSim for every pair of the components `c`, with gamma = gammas[g]."""
    t = title_sim(c, alpha, beta, cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(c["minf"] > 0, c["m"][g] / c["minf"], 0.0)
    if cfg.theta_cap:
        r = np.minimum(1.0, r)
    avg, mwp = c["avg"][g], c["mwp"][g]
    th1 = (1 - mu) * r
    return np.where(t == -1, r * avg + (1 - r) * mwp, th1 * avg + (1 - mu - th1) * mwp + mu * t)


def same_shop(shop):
    """infinite(a, b) of CODE: only products of the same shop are at distance infinity."""
    return lambda a, b: shop[a] == shop[b]


def _merges(n, pairs, dist, max_eps, infinite=None):
    """Adapted single linkage: yields (distance, cluster A, cluster B) for every merge, in order.
    Only `pairs` can trigger a merge; merging stops beyond max_eps. infinite(a, b) tells whether
    two products are at distance infinity (no cluster may contain both); by default every pair
    outside `pairs` is."""
    if infinite is None:
        finite = set(pairs)
        infinite = lambda a, b: ((a, b) if a < b else (b, a)) not in finite
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    below = np.flatnonzero(dist <= max_eps)
    for t in below[np.argsort(dist[below], kind="stable")]:
        ri, rj = find(pairs[t][0]), find(pairs[t][1])
        if ri == rj:
            continue
        A, B = members[ri], members[rj]
        if not any(infinite(a, b) for a in A for b in B):
            yield dist[t], A, B
            parent[rj] = ri
            members[ri] = A + B
            del members[rj]


def cluster(n, pairs, dist, eps, infinite=None):
    """Pairs (i, j), i<j, that end up in the same cluster."""
    out = set()
    for _, A, B in _merges(n, pairs, dist, eps, infinite):
        out.update((a, b) if a < b else (b, a) for a in A for b in B)
    return out


def f1_path(n, pairs, dist, gold, epsilons, infinite=None):
    """F1 of the clustering for every threshold in the ascending list `epsilons` (one pass)."""
    out, tp, npred, k = [], 0, 0, 0
    for d, A, B in _merges(n, pairs, dist, epsilons[-1], infinite):
        while d > epsilons[k]:
            out.append(2 * tp / (npred + len(gold)) if npred + len(gold) else 0.0)
            k += 1
        npred += len(A) * len(B)
        tp += sum(((a, b) if a < b else (b, a)) in gold for a in A for b in B)
    while len(out) < len(epsilons):
        out.append(2 * tp / (npred + len(gold)) if npred + len(gold) else 0.0)
    return out
