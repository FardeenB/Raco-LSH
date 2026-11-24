# RaCo-LSH: Rarity-Weighted LSH with Collision Counting for Scalable Product Duplicate Detection

Code for the paper *Rarity-Weighted LSH with Collision Counting for Scalable Product Duplicate
Detection*. The task is to find product descriptions from different Web shops that refer to the
same product, without comparing all pairs of products.

The data set contains 1,624 televisions from four Web shops (Amazon.com, Newegg.com, Best-Buy.com,
TheNerds.net). They have 1,262 distinct model IDs, which gives 399 duplicate pairs. The model ID is
used **only** as the gold standard. The method never sees it: every experiment passes products
through `common.public()`, which removes it.

## Method

RaCo-LSH changes the three parts of the LSH pre-selection used by earlier work (MSMP, MSMP+), then
classifies only the candidate pairs it keeps.

1. **Typed, schema-agnostic tokens** (`lshdd/tokens.py`). Only the title and the attribute
   *values* are used, never the attribute keys. Each product becomes a set of tokens of five
   classes:

   | Class | Meaning | Examples |
   |---|---|---|
   | `tw` | title words | `samsung`, `plasma` |
   | `tq` | title quantities (number + unit) | `1080p`, `120hz`, `46inch` |
   | `id` | identifier-like title words | `un46es6580f` |
   | `ip` | all proper prefixes (≥ 4 characters) of an identifier | `un46`, `un46e`, … |
   | `vq` | quantities in attribute values, typed by unit; fractions are parsed, and lengths and weights are quantised on two shifted grids | |

2. **Rarity-weighted MinHash** (`lshdd/represent.py`, `lshdd/minhash.py`). Each token gets the
   weight `w_t = idf(t)^γ` with γ = 2. Weighted MinHash then makes the probability that two
   signature entries agree equal to the weighted Jaccard similarity. No labels are needed.

3. **LSH with collision counting** (`lshdd/lsh.py`). 1,200 hash functions are split into b = 400
   bands of r = 3 rows. A pair becomes a candidate when it shares a bucket in **at least k**
   bands; k = 1 is classical banding. Pairs from the same Web shop are skipped.

4. **Matching** (`lshdd/features.py`, `lshdd/pipeline.py`). A random forest with 200 trees uses 17
   cheap pair features, such as the weighted Jaccard similarity per token class, identifier
   relations and unit agreement. It is trained on the candidate pairs of the training products.
   Its decision threshold maximises the out-of-bag F1 on the training set.

The LSH (MinHash, banding, collision counting) is implemented from scratch with numpy. Only the
classifier comes from scikit-learn.

## Baselines

`lshdd/msm.py` and `lshdd/baselines.py` re-implement three earlier methods:
- **MSM** (van Bezu et al., 2015): a multi-component similarity combined with hierarchical
  clustering.
- **MSMP** (van Dam et al., 2016): MSM after LSH on the model words of the title.
- **MSMP+** (Hartveld et al., 2018): MSMP with data cleaning and model words from the attribute
  values.

Where the papers leave details open, the re-implementation follows the authors' public code
(github.com/nickvalstar/APFAProject). The module docstrings list every such choice. The MSM
parameters are chosen by grid search on the training products of each bootstrap.

## Evaluation protocol

- **Bootstrap.** Each bootstrap draws the duplicate clusters with replacement, as the MSM authors'
  code does. The drawn clusters form the training set (on average 63% of the products) and the
  clusters that were never drawn form the test set. A duplicate group is therefore never split
  between training and test.
- **Repetitions.** Every method uses the same 50 bootstraps (seed 2026). All scores are averages
  over the bootstraps, measured on the out-of-bag (test) products.
- **Pre-selection scores:**
  - pair completeness (PC): duplicates found / all duplicates
  - pair quality (PQ): duplicates found / comparisons made
  - F1\*: the harmonic mean of PQ and PC
  - fraction of comparisons: candidate pairs / all n(n−1)/2 pairs of the test set
- **Duplicate detection scores:** precision, recall and F1 over pairs of products.
- **Cleaned gold standard.** Scores marked "clean" compare model IDs without case and punctuation,
  because the gold standard labels pairs such as `LC-60LE650U` / `LC60LE650U` as non-duplicates.

## Repository structure

```
lshdd/                      the method and the baselines (a Python package)
  data.py                   loading the TV data set; gold duplicate pairs
  tokens.py                 typed token extraction (title + attribute values)
  represent.py              vocabulary, document frequencies, IDF^gamma weights
  minhash.py                (weighted) MinHash signatures, own implementation
  lsh.py                    banding and collision counting, own implementation
  features.py               the 17 pair features of the matcher
  pipeline.py               Blocker (LSH pre-selection), Matcher (classifier), DuplicateDetector
  evaluate.py               cluster bootstrap, PC / PQ / F1* and precision / recall / F1
  baselines.py              MSMP and MSMP+ representations and their LSH
  msm.py                    MSM similarity and adapted single-linkage clustering (numba optional)
  brands.py                 brand list used by MSM
experiments/
  common.py                 shared protocol: data, bootstraps, provenance of results files
  run_blocking.py           exp. 1: pre-selection quality vs. fraction of comparisons, ablations
  run_matching.py           exp. 2: end-to-end F1 vs. fraction of comparisons
  run_msm.py                exp. 3: MSM, MSMP and MSMP+ re-implemented end to end
  run_external.py           exp. 4: the pre-selection on Abt-Buy and Amazon-Google
  run_checks.py             leakage check, timings, second independent set of bootstraps
  make_tables.py            prints the tables, writes paper/numbers.tex (LaTeX macros)
  make_figures.py           writes paper/figures/results.pdf / .png
tests/                      unit tests (pytest)
data/
  TVs-all-merged.json       TV data set (https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip)
  external/                 Abt-Buy (D2) and Amazon-Google (D3), as distributed with pyJedAI
                            (https://github.com/AI-team-UoA/pyJedAI, data/ccer)
results/                    results/*.json, written by the experiments
```

## Installation

Use Python 3.10 or newer. The code was tested with Python 3.13, numpy 2.1, scikit-learn 1.7,
matplotlib 3.10 and numba 0.62.

```
pip install numpy scikit-learn matplotlib numba
```

numba is optional. It compiles the clustering used by the MSM baselines. Without numba,
`run_msm.py` gives the same results but is much slower.

## Usage

Run every command from the repository root.

**Run the tests:**

```
python -m pytest tests
```

**Use the method on your own collection of products** (dicts with `shop`, `title` and `features`):

```python
from lshdd.pipeline import DuplicateDetector

det = DuplicateDetector(k=3).fit(train_products, train_gold_pairs)   # gold: set of (i, j), i < j
duplicates, candidates = det.predict(new_products)   # set / sorted list of (i, j) pairs
```

**Reproduce the paper.** The bootstrapped experiments take `[n_bootstraps] [n_processes]`, with
defaults 50 and 4:

```
python experiments/run_blocking.py
python experiments/run_matching.py
python experiments/run_msm.py 50 8
python experiments/run_external.py
python experiments/run_checks.py
python experiments/make_tables.py
python experiments/make_figures.py
```

Notes on these commands:
- `run_checks.py` measures wall-clock times, so run it while nothing else is running.
- Each experiment writes one file to `results/` (`blocking.json`, `matching.json`, `msm.json`,
  `external.json`, `checks.json`). Each row is keyed by method and setting, for example
  `ours|r3|k3` (r = 3, k = 3), `rf|r3|k3` or `msmp|t0.50` (LSH threshold 0.50), and holds the
  means and standard deviations over the bootstraps.
- Each results file also stores its provenance under `_meta`: protocol, number of bootstraps, git
  commit and a hash of the code. `make_tables.py` refuses to combine files made under different
  protocols or with different numbers of bootstraps.
- `RESULTS_DIR=<dir>` sends the results somewhere else.

Approximate running times on a 12-thread laptop:

| Script | Time | Notes |
|---|---|---|
| `run_msm.py` | 1.5–2.5 h | with 8–4 processes; the MSM grid search (7,290 settings × 9 ε for every pre-selection) dominates |
| `run_matching.py` | ~25 min | |
| `run_external.py` | ~20 min | |
| `run_checks.py` | ~7 min | |
| `run_blocking.py` | ~5 min | |

The proposed method itself takes seconds per bootstrap.
