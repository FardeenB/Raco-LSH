"""MSM in CODE mode against the logic of the MSM authors' Java code (msm/MsmSimilarity.java,
msm/TitleAnalyzerImpl.java, msm/CustomClustering.java of github.com/nickvalstar/APFAProject).
Expected values are worked out by hand from the Java source.  usage: python -m pytest tests"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lshdd import msm
from lshdd.brands import title_brands
from lshdd.msm import CODE, PAPER, MSM


def lev(a, b):
    """Plain Levenshtein distance, independent of lshdd.msm."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1][-1]


def avg_lv_sim(pairs):
    """avgLvSim of TitleAnalyzerImpl: (1 - normalised distance), weighted by the summed lengths."""
    num = sum((1 - lev(x, y) / max(len(x), len(y))) * (len(x) + len(y)) for x, y in pairs)
    return num / sum(len(x) + len(y) for x, y in pairs)


def product(title, features=None, shop="a"):
    return {"shop": shop, "title": title, "features": features or {"Size": "1"}}


def components(cfg, p, q, gammas=(0.7,)):
    m = MSM([p, dict(q, shop="b")], cfg=cfg)
    return m.components([(0, 1)], gammas)


# --- strings ---------------------------------------------------------------------------------

def test_qgram_padding_and_similarity():
    # TokeniserQGram3Extended pads with "##" on both sides
    assert sorted(msm.qgrams("ab")) == sorted(["##a", "#ab", "ab#", "b##"])
    # "abc" / "abd": 5 q-grams each, 2 shared -> (10 - 6) / 10
    assert msm.qgram_sim(msm.qgrams("abc"), msm.qgrams("abd")) == pytest.approx(0.4)


def test_sparker_text():
    # ", " ". " a final "." and brackets become spaces; white space is kept
    assert msm.sparker_text("Brand, Model (2012). X.") == "brand model  2012  x "


def test_java_split():
    assert msm.java_split("a  b ") == ["a", "b"]
    assert msm.java_split(" a") == ["", "a"]


def test_model_words_java_regex():
    assert msm._model_words_java('toshiba 32" 720p 60hz led-lcd hdtv 32sl410u') == {'32"', "720p", "60hz", "32sl410u"}
    # "^" inside the Java character class is a literal: no model word in "a^1"
    assert msm._model_words_java("a^1") == set()
    assert msm._model_words_java("20.8lbs") == {"20.8lbs"}


def test_levenshtein():
    assert msm.levenshtein("kitten", "sitting") == 3


def test_title_brands():
    assert title_brands('Samsung 46" LED') == {"samsung"}
    assert title_brands("Insignia 32in LED") == frozenset()        # not in sparker's list
    assert title_brands("LG / Samsung combo") == {"lg", "samsung"}
    assert title_brands("Samsung-UN46") == frozenset()             # only white-space separated words


# --- mwPerc ----------------------------------------------------------------------------------

def test_dice_keeps_duplicates():
    assert msm.dice(["a", "b", "a"], ["a", "c"]) == pytest.approx(3 / 5)
    assert msm.dice([], []) == 0.0


def test_mwperc_dice_over_value_lists():
    # no key matches (gamma 0.99); value model words: [32in, 32in, 5kg] versus [32in]
    p = product("x", {"Width": "32in", "Height": "32in", "Mass": "5kg"})
    q = product("y", {"Depth": "32in"})
    assert components(CODE, p, q, (0.99,))["mwp"][0, 0] == pytest.approx((2 + 1) / (3 + 1))
    assert components(PAPER, p, q, (0.99,))["mwp"][0, 0] == pytest.approx(1 / 2)   # Jaccard of sets


def test_value_model_words_from_raw_value():
    # sparker takes model words from the raw value, so case matters; numbers alone are no model words
    p = product("x", {"Resolution": "1080P", "Count": "3"})
    q = product("y", {"Pixels": "1080p", "Number": "3"})
    assert components(CODE, p, q, (0.99,))["mwp"][0, 0] == 0.0
    assert components(PAPER, p, q, (0.99,))["mwp"][0, 0] == 1.0


# --- TMWM ------------------------------------------------------------------------------------

T2 = "Samsung UN46ES6580 46-Inch 1080p 120Hz 3D HDTV"      # MSMP paper, Table 1: 2 and 3 are duplicates
T3 = 'Samsung 46" 1080p 240Hz LED HDTV UN46ES6580'
T4 = 'Toshiba - 32" / LED / 720p / 60Hz / HDTV'              # 4 and 5 are duplicates
T5 = 'Toshiba 32" 720p 60Hz LED-LCD HDTV 32SL410U'


def test_title_cosine_counts_words():
    # 4 of 7 words shared
    assert components(CODE, product(T2), product(T3))["cos"][0] == pytest.approx(4 / 7)
    # word counts: "/" occurs 4 times in T4 -> 5 / sqrt(23 * 7); a set cosine would ignore that
    assert components(CODE, product(T4), product(T5))["cos"][0] == pytest.approx(5 / math.sqrt(23 * 7))


def test_conflict_rule_rejects_true_duplicate():
    c = components(CODE, product(T2), product(T3))
    assert c["conflict"][0]                       # 120hz / 240hz: same unit, other number
    assert msm.title_sim(c, 0.5, 0.0, CODE)[0] == 1.0      # cosine 4/7 > alpha wins first
    assert msm.title_sim(c, 0.6, 0.0, CODE)[0] == -1.0     # otherwise the conflict rejects it
    assert msm.title_sim(c, 0.6, 0.0, PAPER)[0] != -1.0    # PAPER ignores the rule


def test_tmwm_levenshtein_parts():
    c = components(CODE, product(T4), product(T5))
    A, B = ['32"', "720p", "60hz"], ['32"', "720p", "60hz", "32sl410u"]
    assert not c["conflict"][0]
    # average over all pairs of title model words
    assert c["base"][0] == pytest.approx(avg_lv_sim([(x, y) for x in A for y in B]))
    # pairs with equal letters and approximately equal numbers: the three identical words
    assert c["mwsim"][0] == pytest.approx(1.0)
    expected = 0.5 * 1.0 + 0.5 * (0.5 * c["cos"][0] + 0.5 * c["base"][0])
    assert msm.title_sim(c, 0.9, 0.0, CODE)[0] == pytest.approx(expected)
    assert msm.title_sim(c, 0.9, 0.95, CODE)[0] == -1.0    # not above beta


def test_tmwm_without_model_words_is_rejected():
    # no model words: avgLvSim is 0/0 = NaN in Java, so TMWM returns -1 unless the cosine decides
    c = components(CODE, product("Samsung LED HDTV"), product("Samsung Plasma HDTV"))
    assert c["cos"][0] == pytest.approx(2 / 3)
    assert msm.title_sim(c, 0.7, 0.0, CODE)[0] == -1.0
    assert msm.title_sim(c, 0.6, 0.0, CODE)[0] == 1.0
    assert msm.title_sim(components(PAPER, product("Samsung LED HDTV"), product("Samsung Plasma HDTV")),
                         0.7, 0.0, PAPER)[0] > 0


# --- hSim, brands, clustering ----------------------------------------------------------------

def _c(m, minf, **kw):
    c = dict(avg=np.array([[0.5]]), m=np.array([[m]]), mwp=np.array([[0.2]]), minf=np.array([minf]),
             cos=np.array([0.0]), conflict=np.array([False]), base=np.array([np.nan]), mwsim=np.array([np.nan]))
    c.update(kw)
    return c


def test_theta_not_capped():
    # 3 key matches, 2 features: th1 = 1.5 in Java; title rejected (NaN) -> th1 avg + (1 - th1) mwPerc
    assert msm.hsim(_c(3, 2), 0, 0.5, 0.0, 0.5, CODE)[0] == pytest.approx(1.5 * 0.5 - 0.5 * 0.2)
    capped = _c(3, 2, base=np.array([0.0]))                  # PAPER: th1 capped at 1
    t = msm.title_sim(capped, 0.5, 0.0, PAPER)[0]
    assert t == -1.0
    assert msm.hsim(capped, 0, 0.5, 0.0, 0.5, PAPER)[0] == pytest.approx(0.5)


def test_different_brands_are_not_scored():
    prods = [product("Samsung 32in", shop="a"), product("LG 32in", shop="b"), product("TV 32in", shop="c")]
    assert MSM(prods, cfg=CODE).valid_pairs() == [(0, 2), (1, 2)]


def test_unscored_pairs_do_not_block_in_code_mode():
    # three shops; (0, 2) is not scored. Java: distance 1 -> the chain 0-1-2 is merged;
    # PAPER: distance infinity -> 2 cannot join the cluster of 0 and 1
    pairs, d = [(0, 1), (1, 2)], np.array([0.1, 0.2])
    code = msm.cluster(3, pairs, d, 0.5, msm.same_shop(["a", "b", "c"]))
    assert code == {(0, 1), (0, 2), (1, 2)}
    assert msm.cluster(3, pairs, d, 0.5) == {(0, 1)}


def test_same_shop_blocks_in_code_mode():
    # 0 and 2 come from the same shop: never in one cluster
    pairs, d = [(0, 1), (1, 2)], np.array([0.1, 0.2])
    assert msm.cluster(3, pairs, d, 0.5, msm.same_shop(["a", "b", "a"])) == {(0, 1)}
