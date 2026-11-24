"""Brands for the diffBrand() rule of MSM (van Bezu et al., SAC 2015), as in sparker, the MSM authors'
code. Only used by the MSM / MSMP / MSMP+ baselines; our own method does not use brands."""
import re

# Brand list of sparker (TitleAnalyzerImpl.java in github.com/nickvalstar/APFAProject); "sony" is
# listed twice there.
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
