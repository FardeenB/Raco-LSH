import os, sys, re, json
from collections import defaultdict
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from lshdd.data import load_products
from lshdd.evaluate import bootstrap_splits, local_gold, blocking_scores, prf

DATA = os.environ.get("TV_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "data", "TVs-all-merged.json"))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SPLIT_SEED = 2026


def splits(n, n_boot):
    return list(bootstrap_splits(n, n_boot, seed=SPLIT_SEED))


def cleaned_gold(products, idx):
    """Gold pairs when model IDs are compared case-insensitively and without punctuation
    (e.g. 'LC-60LE650U' == 'LC60LE650U'). Only used for the secondary 'cleaned gold' scores."""
    groups = defaultdict(list)
    for pos, g in enumerate(idx):
        groups[re.sub(r'[^a-z0-9]', '', products[int(g)]["gold"].lower())].append(pos)
    return {pr for v in groups.values() for pr in combinations(v, 2)}


def public(p):
    """Copy of a product WITHOUT the gold model ID: all that the method is allowed to see."""
    return {"shop": p["shop"], "title": p["title"], "features": p["features"]}


def mean_rows(rows):
    keys = rows[0].keys()
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out.update({k + "_sd": float(np.std([r[k] for r in rows])) for k in keys})
    out["n_boot"] = len(rows)
    return out


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)
