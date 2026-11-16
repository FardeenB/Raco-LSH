"""Loading of the TV data set and gold-standard helpers."""
import json
from itertools import combinations
from collections import defaultdict


def load_products(path):
    """Return a flat list of products. The modelID is kept ONLY as the gold label."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    products = []
    for model_id, group in raw.items():
        for p in group:
            products.append({
                "shop": p["shop"],
                "title": p["title"],
                "features": p["featuresMap"],
                "gold": model_id,          # never read by any representation / matcher
            })
    return products


def gold_pairs(products, idx=None):
    """Set of (i, j), i<j, of true duplicate pairs among the given indices."""
    if idx is None:
        idx = range(len(products))
    groups = defaultdict(list)
    for i in idx:
        groups[products[i]["gold"]].append(i)
    out = set()
    for g in groups.values():
        for a, b in combinations(sorted(g), 2):
            out.add((a, b))
    return out
