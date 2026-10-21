"""Formulary classification rules — pure functions, no file reading.

Everything the build decides about one sales line lives here, so it can be unit-tested
without HCG's workbooks:

  category(code)          first digit of MATERIALCODE: 1 Onco Drugs, 2 Other Drugs,
                          3 Consumables. NOT HCG's 'Onco/Non Onco' column: for Aug-26 it
                          labels 151 onco codes (4,310 lines) as Non Onco. NOT the Item
                          Master's Material type either (HCG does not use it since Jul-26).
  formulary_class(flag, cat)
                          the same month's Item Master: FORMULARY + Category, NON
                          FORMULARY, OUT OF FORMULARY, or NOT IN MASTER. Reproduces HCG's
                          'FY 26-27 - CATEGORY' column on 100% of lines in Jun, Jul and Aug.
  formulary_group(cls)    A formulary (incl. FORMULARYCONSUMABLE, HCG's rule), B out of
                          formulary, C non formulary, X not classified.
  payer_bucket(...)       CASH (blank payer), else payer_overrides.csv, else the Payer
                          Master's 'Final after Discussion' value of the FIRST row with that
                          CUSTOMERNAME (HCG's lookup), else CASH(TPA) and reported.
  pooled_pct / mean_pct   compliance = A / (A + B); C is excluded. Always a ratio of sums.

The category names are the app's own (app/core/data_access.CATEGORIES), so the existing
Category filter and resolve_category keep working on the formulary tables.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Sequence

ONCO = "Onco Drugs"
OTHER = "Other Drugs"
CONSUMABLES = "Consumables"
UNCLASSIFIED = "Unclassified"
DRUG_CATEGORIES = (ONCO, OTHER)
CATEGORY_BY_DIGIT = {"1": ONCO, "2": OTHER, "3": CONSUMABLES}
# Display labels, and the API's Segment keys for the two drug categories.
SEGMENT_LABEL = {ONCO: "Onco", OTHER: "Non-Onco", CONSUMABLES: "Consumables"}
SEGMENT_KEY = {ONCO: "onco", OTHER: "non-onco"}

FORMULARY = "FORMULARY"
NON_FORMULARY = "NON FORMULARY"
OUT_OF_FORMULARY = "OUT OF FORMULARY"
NOT_IN_MASTER = "NOT IN MASTER"
# Category spellings HCG fixed between masters (Jun 'OPD-R' became Jul 'R-OPD'; 'P-0PD'
# has a zero). Canonicalised so a class means the same thing in every month.
CATEGORY_TYPOS = {"OPD-R": "R-OPD", "P-0PD": "P-OPD"}
# Scheme views: restricted / preferred brands as billed to GOV payers.
RESTRICTED_CLASSES = ("FORMULARYR", "FORMULARYR-OPD")
PREFERRED_CLASSES = ("FORMULARYP",)
# HCG's Category letters on FORMULARY items (after CATEGORY_TYPOS), and what a brand rule
# can say about a class x payer x patient type: a breach of HCG's guidelines, a figure HCG
# watches, or information until HCG confirms the rule.
LETTERS = ("P", "P-SCH", "SCH", "R", "R-OPD", "OPD", "P-OPD", "SCH-OPD", "CONSUMABLE")
VERDICTS = ("breach", "watch", "info")

CASH, CASH_TPA, GOV = "CASH", "CASH(TPA)", "GOV"
PAYER_BUCKETS = (CASH, CASH_TPA, GOV)

# Excel error cells as the two readers return them: openpyxl gives the error text,
# pyxlsb gives the BIFF12 error byte in hex ('0x2a' is #N/A). Plus HCG's own 'N/A'.
EXCEL_ERRORS = frozenset({
    "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!", "#GETTING_DATA",
    "0X0", "0X7", "0XF", "0X17", "0X1D", "0X24", "0X2A", "0X2B", "N/A",
})

_WS = re.compile(r"\s+")           # str patterns: \s includes the non-breaking space
_TRAILING_ZEROS = re.compile(r"\.0+$")
_PERIOD = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def norm_text(v) -> str:
    """Collapse whitespace (incl. non-breaking spaces), trim, upper. '' for None."""
    if v is None:
        return ""
    return _WS.sub(" ", str(v)).strip().upper()


def is_error_value(v) -> bool:
    return isinstance(v, str) and v.strip().upper() in EXCEL_ERRORS


def is_blank(v) -> bool:
    """Blank in any of the ways the HIS and HCG write it: None, '', spaces, 'Blank',
    'Blank ' — and Excel error cells."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    s = norm_text(v)
    return s in ("", "BLANK") or s in EXCEL_ERRORS


def norm_code(v) -> Optional[str]:
    """Material / doctor codes: 101313.0, '101313.0', ' 101313 ' -> '101313'."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        if v != v:
            return None
        if v.is_integer():
            v = int(v)
    s = _TRAILING_ZEROS.sub("", str(v).strip())
    if not s or is_error_value(s):
        return None
    return s


def category(code) -> str:
    c = norm_code(code)
    return CATEGORY_BY_DIGIT.get(c[:1], UNCLASSIFIED) if c else UNCLASSIFIED


def is_drug(cat: Optional[str]) -> bool:
    return cat in DRUG_CATEGORIES


def segment_label(cat: Optional[str]) -> Optional[str]:
    return SEGMENT_LABEL.get(cat)


def canonical_category(cat) -> str:
    c = norm_text(cat)
    return CATEGORY_TYPOS.get(c, c)


def formulary_class(flag, cat=None) -> str:
    """Class of an item from its master row. flag None = code not in that month's master."""
    if flag is None:
        return NOT_IN_MASTER
    f = norm_text(flag)
    if f == FORMULARY:
        return FORMULARY + canonical_category(cat)
    if f in (NON_FORMULARY, OUT_OF_FORMULARY):
        return f
    raise ValueError(f"unknown formulary flag {flag!r}")


def rule_class(cls) -> Optional[str]:
    """A class as a brand-rules file may write it -> the published formulary_class, or None
    when it is not one: 'FORMULARYR-OPD', 'formularyopd-r' -> 'FORMULARYR-OPD'."""
    c = norm_text(cls)
    if c in (OUT_OF_FORMULARY, NON_FORMULARY):
        return c
    if c.startswith(FORMULARY):
        letter = canonical_category(c[len(FORMULARY):])
        return FORMULARY + letter if letter in LETTERS else None
    return None


def rule_letter(cls) -> str:
    """'FORMULARYR-OPD' -> 'R-OPD'; 'OUT OF FORMULARY' / 'NON FORMULARY' unchanged."""
    c = norm_text(cls)
    return c[len(FORMULARY):] if c.startswith(FORMULARY) else c


def formulary_group(cls: Optional[str]) -> str:
    c = norm_text(cls)
    if c.startswith(FORMULARY):
        return "A"
    if c == OUT_OF_FORMULARY:
        return "B"
    if c == NON_FORMULARY:
        return "C"
    return "X"


# --------------------------------------------------------------------------- #
# HCG's worksheet columns, normalised for the agreement check only
# --------------------------------------------------------------------------- #
def hcg_class(v) -> str:
    """HCG's 'FY 26-27 - CATEGORY' value in our class vocabulary."""
    s = norm_text(v)
    if not s or s in EXCEL_ERRORS:
        return NOT_IN_MASTER
    if s.startswith(FORMULARY):
        return FORMULARY + canonical_category(s[len(FORMULARY):])
    return s


def hcg_segment(v) -> Optional[str]:
    """HCG's 'Onco/Non Onco' ('A-Onco', 'B-Non Onco', 'C-Consumables', June's
    'C-Consumbales') as our segment label."""
    s = norm_text(v).replace("-", " ")
    if not s or s in EXCEL_ERRORS:
        return None
    if "NON ONCO" in s:
        return SEGMENT_LABEL[OTHER]
    if "ONCO" in s:
        return SEGMENT_LABEL[ONCO]
    if "CONSUM" in s:
        return SEGMENT_LABEL[CONSUMABLES]
    return None


def norm_bucket(v) -> Optional[str]:
    """'Cash (TPA)', 'gov ' -> 'CASH(TPA)', 'GOV'. None when not one of the three."""
    s = _WS.sub("", norm_text(v))
    return s if s in PAYER_BUCKETS else None


# --------------------------------------------------------------------------- #
# Payer bucket
# --------------------------------------------------------------------------- #
def payer_key(v) -> str:
    """Lookup key for a payer name. Punctuation is kept on purpose: stripping it merges
    'Central Government Health Scheme-CGHS' (GOV) into '...(CGHS)' (CASH(TPA))."""
    return norm_text(v)


def is_period(s) -> bool:
    return isinstance(s, str) and bool(_PERIOD.match(s))


def override_bucket(key: str, period: str,
                    overrides: Mapping[str, Sequence]) -> Optional[str]:
    """overrides: key -> [(from_period or None, to_period or None, bucket), ...]."""
    for lo, hi, bucket in overrides.get(key, ()):
        if (lo is None or period >= lo) and (hi is None or period <= hi):
            return bucket
    return None


def resolve_payer(payer1, period: str, master: Optional[Mapping[str, str]] = None,
                  overrides: Optional[Mapping[str, Sequence]] = None) -> tuple:
    """(bucket, how) for one PAYER1 value. how: blank | override | master | unmatched."""
    if is_blank(payer1):
        return CASH, "blank"
    k = payer_key(payer1)
    b = override_bucket(k, period, overrides or {})
    if b:
        return b, "override"
    b = (master or {}).get(k)
    if b:
        return b, "master"
    return CASH_TPA, "unmatched"


def payer_bucket(payer1, period: str, master: Optional[Mapping[str, str]] = None,
                 overrides: Optional[Mapping[str, Sequence]] = None) -> str:
    return resolve_payer(payer1, period, master, overrides)[0]


def first_row_buckets(rows: Iterable) -> tuple:
    """Payer Master (CUSTOMERNAME, Final bucket) pairs in SHEET ORDER -> the name map
    HCG's lookup produces (the first row wins), plus the names whose rows disagree.

    The first-row rule is HCG's, not ours: 'Central Government Health Scheme (CGHS)' is
    CASH(TPA) on its first row (DR) and GOV on its KR and Mumbai rows, so HCG's column
    says CASH(TPA) at KR. The conflicts are reported so they can be settled with HCG."""
    names: dict = {}
    seen: dict = {}
    for name, bucket in rows:
        k = payer_key(name)
        if not k or bucket is None:
            continue
        names.setdefault(k, bucket)
        counts = seen.setdefault(k, {})
        counts[bucket] = counts.get(bucket, 0) + 1
    conflicts = [{"name": k, "used": names[k], "rows_by_bucket": v}
                 for k, v in seen.items() if len(v) > 1]
    return names, conflicts


# --------------------------------------------------------------------------- #
# Compliance — always a ratio of sums, never an average of percentages
# --------------------------------------------------------------------------- #
def pct(num, den) -> Optional[float]:
    if den is None or den == 0 or num is None:
        return None
    return float(num) / float(den) * 100.0


def pooled_pct(formulary, out_of_formulary) -> Optional[float]:
    """A / (A + B) * 100. Non-formulary (C) and consumables are not in it."""
    return pct(formulary, (formulary or 0) + (out_of_formulary or 0))


def mean_pct(onco_a, onco_b, non_onco_a, non_onco_b) -> Optional[float]:
    """HCG's method before Jul-26: the average of the onco and the non-onco ratio. A
    segment with A + B = 0 has no ratio and is left out; None when neither has one."""
    vals = [p for p in (pooled_pct(onco_a, onco_b), pooled_pct(non_onco_a, non_onco_b))
            if p is not None]
    return sum(vals) / len(vals) if vals else None
