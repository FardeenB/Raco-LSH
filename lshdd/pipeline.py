"""End-to-end duplicate detection: typed tokens -> IDF-weighted MinHash -> (b, r, k) LSH ->
light supervised matcher on the candidate pairs only."""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .tokens import product_tokens
from .represent import Vocabulary, idf_weights
from .minhash import signatures, token_codes
from .lsh import collision_counts
from .features import PairFeaturizer


class Blocker:
    """Candidate generation. Unsupervised: it only sees the collection to be de-duplicated."""

    def __init__(self, gamma=2.0, n_hashes=1200, rows=3, weighted=True, seed=7, tokenizer=None):
        self.gamma, self.n_hashes, self.rows = gamma, n_hashes, rows
        self.weighted, self.seed = weighted, seed
        self.tokenizer = tokenizer or product_tokens

    @property
    def bands(self):
        return self.n_hashes // self.rows

    def index(self, products):
        """Returns a dict with token sets, vocabulary, weights and band-collision counts."""
        t0 = time.perf_counter()
        toks = [self.tokenizer(p) for p in products]
        vocab = Vocabulary(toks)
        w = idf_weights(vocab, self.gamma) if self.weighted else np.ones(len(vocab.tokens))
        sig = signatures(vocab.ids, token_codes(vocab.tokens), w if self.weighted else None,
                         self.bands * self.rows, seed=self.seed)
        shops = np.array([p["shop"] for p in products])
        counts = collision_counts(sig, self.bands, self.rows, block=shops)
        return dict(products=products, tokens=toks, vocab=vocab, weights=w, counts=counts,
                    seconds=time.perf_counter() - t0)

    @staticmethod
    def candidates(index, k):
        return sorted(p for p, c in index["counts"].items() if c >= k)


def _best_threshold(scores, labels, n_gold):
    """Threshold maximising F1, where recall is measured against ALL gold pairs."""
    if len(scores) == 0 or labels.sum() == 0:
        return 0.5
    order = np.argsort(-scores)
    s = scores[order]
    tp = np.cumsum(labels[order])
    f1 = 2 * tp / (np.arange(1, len(s) + 1) + n_gold)
    valid = np.r_[s[1:] != s[:-1], True]
    f1 = np.where(valid, f1, -1.0)
    i = int(f1.argmax())
    nxt = s[i + 1] if i + 1 < len(s) else 0.0
    return float((s[i] + nxt) / 2)


class Matcher:
    """Binary classifier on cheap pair features, trained on LSH candidates of the training set."""

    def __init__(self, model="rf", seed=0):
        self.model_name, self.seed = model, seed

    def _make(self):
        if self.model_name == "rf":
            return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, oob_score=True,
                                          n_jobs=1, random_state=self.seed)
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))

    def fit(self, X, y, n_gold):
        self.constant = None
        if len(y) == 0 or y.all() or not y.any():
            self.constant = bool(len(y) and y.all())
            return self
        self.clf = self._make()
        if self.model_name == "rf":
            self.clf.fit(X, y)
            oof = self.clf.oob_decision_function_[:, 1]
            oof = np.where(np.isnan(oof), 0.0, oof)
        else:
            cv = min(5, int(y.sum()), int((~y).sum()))
            if cv >= 2:
                oof = cross_val_predict(self._make(), X, y, cv=cv, method="predict_proba")[:, 1]
            else:
                oof = None
            self.clf.fit(X, y)
            if oof is None:
                oof = self.clf.predict_proba(X)[:, 1]
        self.threshold = _best_threshold(oof, y, n_gold)
        return self

    def scores(self, X):
        if self.constant is not None:
            return np.full(len(X), 1.0 if self.constant else 0.0)
        if len(X) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return self.scores(X) >= self.threshold


def featurize(index, pairs):
    f = PairFeaturizer(index["products"], index["tokens"], index["vocab"], index["weights"])
    return f.matrix(pairs)


class DuplicateDetector:
    """fit() on labelled training products, predict() on a new collection."""

    def __init__(self, k=3, model="rf", **blocker_args):
        self.k = k
        self.blocker = Blocker(**blocker_args)
        self.matcher = Matcher(model)

    def fit(self, products, gold):
        """gold: set of (i, j), i<j, of true duplicate pairs among `products`."""
        idx = self.blocker.index(products)
        pairs = self.blocker.candidates(idx, self.k)
        X = featurize(idx, pairs)
        y = np.array([p in gold for p in pairs], dtype=bool)
        self.matcher.fit(X, y, len(gold))
        return self

    def predict(self, products):
        idx = self.blocker.index(products)
        pairs = self.blocker.candidates(idx, self.k)
        keep = self.matcher.predict(featurize(idx, pairs))
        return {p for p, m in zip(pairs, keep) if m}, pairs
