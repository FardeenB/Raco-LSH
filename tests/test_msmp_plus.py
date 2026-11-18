"""MSMP+ representation against its paper (Hartveld et al., CAiSE 2018): the data cleaning of
Table 1 and the three steps of Fig. 2-4 (clean, values, MSMP+).  usage: python -m pytest tests"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lshdd.baselines import _clean, msmp_plus_tokenizer

# Table 1: every listed representation becomes the normalised unit; "23 Inch" -> "23inch" is the
# paper's own example of removing the space in front of the unit
TABLE_1 = ["Inch", "inches", '"', "-inch", " inch", "inch"], ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", "hz"]


@pytest.mark.parametrize("unit", TABLE_1[0])
def test_inch(unit):
    assert _clean("23" + unit) == "23inch"


@pytest.mark.parametrize("unit", TABLE_1[1])
def test_hertz(unit):
    assert _clean("120" + unit) == "120hz"


def test_cleaning_examples():
    assert _clean("23 Inch") == "23inch"
    assert _clean("23”") == "23inch"                     # typographic double quote
    assert _clean('Samsung 46" Class 4512" Diag. 1080p 120Hz') == "samsung 46inch class 4512inch diag. 1080p 120hz"


P1 = {"shop": "a", "title": 'Samsung 46" 120Hz LED',
      "features": {"Screen Size": "46 Inch", "Weight": "20.8 lbs", "Refresh": "120 Hz", "Inputs": "3"}}
P2 = {"shop": "b", "title": 'LG 32" TV', "features": {"Size": "32inch", "Refresh Rate": "120Hz", "Weight": "20.8lbs"}}


def tokens(**kw):
    tok = msmp_plus_tokenizer([P1, P2], **kw)
    return tok(P1), tok(P2)


def test_clean():
    # MSMP on the cleaned title only
    assert tokens(title_mw_in_values=False, value_numbers=False) == (
        {("mw", "46inch"), ("mw", "120hz")}, {("mw", "32inch")})


def test_clean_values():
    # title model words of the collection that re-occur in the values: 120hz comes from P1's title
    assert tokens(value_numbers=False)[1] == {("mw", "32inch"), ("mw", "120hz")}


def test_msmp_plus_decimal_numbers():
    # numeric part of decimal value model words, alone or with a unit ("20.8", "20.8lbs");
    # integers ("3", the 120 of "120hz") are left out
    p1, p2 = tokens()
    assert p1 == {("mw", "46inch"), ("mw", "120hz"), ("num", "20.8")}
    assert p2 == {("mw", "32inch"), ("mw", "120hz"), ("num", "20.8")}


def test_literal_regex_admits_integers():
    p1, _ = tokens(decimals_only=False)
    assert {("num", "3"), ("num", "120"), ("num", "46")} <= p1
