"""Manually maintained list of TV brands, as used by the diffBrand() rule of MSM (van Bezu et al.,
SAC 2015). Only used by the MSM / MSMP / MSMP+ baselines; our own method does not use brands."""
import re

BRANDS = (
    "samsung", "lg", "panasonic", "toshiba", "sharp", "sony", "vizio", "philips", "coby", "rca",
    "haier", "insignia", "sansui", "dynex", "nec", "westinghouse", "sunbritetv", "tcl", "jvc",
    "viewsonic", "supersonic", "magnavox", "sceptre", "naxa", "contex", "sanyo", "proscan",
    "hisense", "pyle", "mitsubishi", "hannspree", "seiki", "venturer", "upstar", "affinity", "avue",
    "optoma", "curtis young", "curtisyoung", "compaq", "azend", "hiteker", "epson", "hp", "elo",
    "hello kitty", "gpx", "viore", "sigmac", "craig", "pioneer", "hitachi", "funai", "emerson",
    "sylvania", "polaroid", "element", "apex", "zenith", "akai", "jensen", "planar",
)
_CANON = {"curtisyoung": "curtis young"}
_BRAND = re.compile(r"\b(%s)\b" % "|".join(re.escape(b) for b in sorted(BRANDS, key=len, reverse=True)))


def find_brand(text):
    """Left-most brand mentioned in `text` ('Sharp Elite ...' -> 'sharp'), or None."""
    m = _BRAND.search(text.lower())
    return _CANON.get(m.group(1), m.group(1)) if m else None


def brand_of(product):
    """Brand from the title; if absent, from an attribute whose key mentions the brand."""
    b = find_brand(product["title"])
    if b is None:
        for k, v in product["features"].items():
            if "brand" in k.lower() or "manufacturer" in k.lower():
                b = find_brand(v)
                if b:
                    break
    return b


# Brand list of sparker, the MSM authors' code (TitleAnalyzerImpl.java in
# github.com/nickvalstar/APFAProject); "sony" is listed twice there.
SPARKER_BRANDS = frozenset((
    "samsung", "philips", "sony", "sharp", "nec", "hp", "toshiba", "hisense", "lg", "sanyo", "coby",
    "panasonic", "sansui", "vizio", "viewsonic", "sunbritetv", "haier", "optoma", "proscan", "jvc", "pyle",
    "sceptre", "magnavox", "mitsubishi", "supersonic", "compaq", "hannspree", "upstar", "seiki", "rca",
    "craig", "affinity", "naxa", "westinghouse", "epson", "elo", "sigmac",
))


def title_brands(title):
    """Brands of SPARKER_BRANDS among the white-space separated words of the lower-cased title
    (differentTitleBrands: two products differ when both have such brands but none in common)."""
    return frozenset(re.split(r"[ \t\n\x0b\f\r]+", title.lower())) & SPARKER_BRANDS
