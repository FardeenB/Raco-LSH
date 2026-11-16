"""Schema-agnostic, typed token extraction for product descriptions.

Every product is turned into a set of (class, token) pairs. No attribute keys are used;
only the title and the *values* of the key-value pairs, so no schema alignment is needed.

Token classes
    tw   title word (alphabetic)
    tq   title quantity: number + unit (e.g. 1080p, 120hz, 46inch)
    tn   title bare number
    id   identifier-like token (letters and digits mixed, not a quantity), from the title
    ip   proper prefix of an identifier-like title token (prefix expansion)
    vi   identifier-like token found in an attribute value
    vq   quantity found in an attribute value, typed by unit and quantised (fractions parsed)
"""
import re
import html
from fractions import Fraction

_INCH = re.compile(r'(?<=\d)\s*(?:-\s*)?(?:"|”|″|\'\'|inches\b|inch\b|in\.)')
_HZ = re.compile(r'(?<=\d)\s*-?\s*(?:hz|hertz)\b')
_LB = re.compile(r'(?<=\d)\s*(?:lbs?\.?|pounds?)(?![a-z])')
_SPLIT = re.compile(r'[\s/,;:()\[\]|+*]+')
_QTY = re.compile(r'^\d+(?:\.\d+)?[a-z]+$')
_NUM = re.compile(r'^\d+(?:\.\d+)?$')

# number (optionally a mixed fraction such as 13-7/8 or 45 2/7) followed by an optional unit
_MEAS = re.compile(
    r'(?<![a-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:[\s-](\d{1,2})/(\d{1,3}))?\s*'
    r'(inch|lb|kg|mm|cm|hz|w\b|watts?\b|kwh|v\b|ms\b|nits?\b|:1|p\b|i\b|d\b)?')

_UNIT = {"watt": "w", "watts": "w", "nits": "nit"}


def normalise(text):
    s = html.unescape(text).lower()
    s = _INCH.sub("inch ", s)
    s = _HZ.sub("hz ", s)
    s = _LB.sub("lb ", s)
    return s


def _is_mixed(t):
    return any(c.isdigit() for c in t) and any(c.isalpha() for c in t)


def _word_tokens(s):
    for raw in _SPLIT.split(s):
        raw = raw.strip("-.'\"&#!?%$")
        if not raw:
            continue
        if _is_mixed(raw):
            t = re.sub(r'[^a-z0-9.]', '', raw)
            if not _QTY.match(t):
                t = t.replace('.', '')
            yield t
        else:
            for part in raw.split('-'):
                part = part.strip(".'\"")
                if part:
                    yield part


def title_tokens(title, min_prefix=4, prefixes=True):
    out = set()
    for t in _word_tokens(normalise(title)):
        if _NUM.match(t):
            out.add(("tn", t))
        elif _QTY.match(t):
            out.add(("tq", t))
        elif _is_mixed(t) and len(t) >= 4:
            out.add(("id", t))
            if prefixes:
                for k in range(min_prefix, len(t)):
                    out.add(("ip", t[:k]))
        elif t.isalpha() or _is_mixed(t):
            out.add(("tw", t))
    return out


def _quantise(x, unit):
    """Two overlapping grids so that close measurements share at least one token."""
    if unit == "inch":
        w = 0.5
    elif unit in ("lb", "kg"):
        w = 0.2
    else:
        return [repr(round(x, 2))]
    a = int(x / w)
    b = int(x / w + 0.5)
    return ["a%d" % a, "b%d" % b]


def value_tokens(features, ids=True):
    out = set()
    for v in features.values():
        s = normalise(v)
        for m in _MEAS.finditer(s):
            whole, dec, num, den, unit = m.groups()
            if unit is None:
                continue
            x = float(whole.replace(",", "") + ("." + dec if dec else ""))
            if num and den and int(den) != 0 and not dec:
                x += float(Fraction(int(num), int(den)))
            unit = _UNIT.get(unit, unit)
            for q in _quantise(x, unit):
                out.add(("vq", unit + q))
        if ids:
            for t in _word_tokens(s):
                if _is_mixed(t) and len(t) >= 5 and not _QTY.match(t) and '.' not in t:
                    out.add(("vi", t))
    return out


DEFAULT_CLASSES = ("tw", "tq", "id", "ip", "vq")


def product_tokens(p, classes=DEFAULT_CLASSES):
    toks = title_tokens(p["title"]) | value_tokens(p["features"], ids="vi" in classes)
    return {t for t in toks if t[0] in classes}
