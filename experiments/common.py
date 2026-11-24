"""Shared set-up of the experiments: data, evaluation protocol and results files.

Protocol of every experiment: N_BOOT bootstraps over the duplicate clusters (get_splits), the
same draws in every experiment; parameters are only ever chosen on the training products.
RESULTS_DIR (environment) redirects results/*.json. Every results file stores the protocol and the
code it was made with under the key "_meta".
"""
import os, sys, re, json, hashlib, subprocess, datetime
from collections import defaultdict
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from lshdd.data import load_products
from lshdd.evaluate import cluster_bootstrap_splits, local_gold, blocking_scores, prf

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data", "TVs-all-merged.json")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
SPLIT = "cluster"
SPLIT_SEED = 2026
N_BOOT = 50


def get_splits(products, n_boot, seed=SPLIT_SEED):
    """(train_idx, test_idx) of every bootstrap: duplicate clusters drawn with replacement from one
    pool, as in sparker, the MSM authors' code (evaluate.cluster_bootstrap_splits). Bootstrap i is
    the same draw for every n_boot."""
    return list(cluster_bootstrap_splits([p["gold"] for p in products], n_boot, seed=seed))


def cli_args(default_procs=4):
    """(n_bootstraps, n_processes) from the command line: [n_bootstraps] [n_processes]."""
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else N_BOOT
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else default_procs
    return nb, nproc


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


def _git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hash():
    """Hash of every Python file of the repository that is loaded in this process, so that two
    results files can be compared even when they were made from an uncommitted working tree."""
    files = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                    if getattr(m, "__file__", None) and m.__file__.endswith(".py")
                    and os.path.abspath(m.__file__).startswith(ROOT + os.sep)})
    h = hashlib.sha1()
    for f in files:
        h.update(os.path.relpath(f, ROOT).replace(os.sep, "/").encode())
        with open(f, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def meta(script, n_boot=None, **extra):
    """Provenance of a results file. n_boot=None for experiments without bootstraps."""
    out = dict(script=script, n_boot=n_boot,
               split=SPLIT if n_boot else None, split_seed=SPLIT_SEED if n_boot else None,
               git_commit=_git("rev-parse", "--short", "HEAD"),
               git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
               code_hash=code_hash(), date=datetime.datetime.now().isoformat(timespec="seconds"))
    out.update(extra)
    return out


def save(name, obj, meta):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump({"_meta": meta, **obj}, f, indent=1)

