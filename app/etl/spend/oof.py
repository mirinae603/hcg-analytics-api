"""Shared description parsers: what a drug's material description and Item Master 'Minor
Group' say about the product, as comparable keys.

  strength_key   the doses written in the description in one unit ('2 GM / 0.5 MG' and
                 '2000MG/0.5MG' agree), plus the volume when the dose alone does not fix the pack
  form_key       the dosage form: injection, tablet, capsule, dry-powder / metered-dose inhaler,
                 nebuliser, oral liquid, ...
  kind_key       modified release (SR, ER, XR, ...), liposomal / pegylated or protein-bound: kinds
                 of a molecule that are not substitutes for the plain one

Two products with the same molecule, strength, form and kind are the same medicine from
different makers. The purchase build (app/etl/purchase) reads the molecule from the Item
Master 'Minor Group' (purchase/molecule.py) and uses these keys for its out-of-formulary
fallback rule, price variation and generic vs branded views. The out-of-formulary
margin itself now follows HCG's purchase (GRN) method in app/etl/purchase; the sales-based
estimate this module used to make (oof_switch) is retired.
"""
from __future__ import annotations

import re

from app.etl.formulary import classify as K

# Dose units -> (canonical unit, factor). Volumes are a separate kind (see strength_key).
_DOSE = {"MCG": ("MG", 0.001), "MG": ("MG", 1.0), "GM": ("MG", 1000.0), "G": ("MG", 1000.0),
         "IU": ("IU", 1.0), "MIU": ("IU", 1e6), "KIU": ("IU", 1e3), "U": ("IU", 1.0), "UNIT": ("IU", 1.0),
         "UNITS": ("IU", 1.0), "UNT": ("IU", 1.0), "MEQ": ("MEQ", 1.0), "MMOL": ("MMOL", 1.0),
         "%": ("%", 1.0)}
_VOLUME = {"ML": ("ML", 1.0), "L": ("ML", 1000.0), "LTR": ("ML", 1000.0)}
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(UNITS|UNIT|UNT|MCG|MMOL|MEQ|MIU|KIU|MG|GM|IU|ML|LTR|%|G|L|U)(?![A-Z])")
# A dose per mL with no volume in the ratio ('100IU/ML', '2 MG / ML'): a concentration.
_PER_ML = re.compile(r"(MCG|MG|GM|IU|U|UNITS?|MEQ|MMOL)\s*/\s*ML\b")
# A number standing alone ('DOLO 650 TAB'): not part of a word ('B12', '60K') or a pack count ("10'S").
_BARE = re.compile(r"(?<![A-Z\d.])(\d+(?:\.\d+)?)(?![\dA-Z.']|\s*'S)")

# Dosage form from the description: first rule that matches. Inhalers are three forms: one
# dry-powder capsule, one metered-dose canister and one nebuliser ampoule are different units.
FORMS = (
    ("DPI", r"ROTACAP|\bDPI\b|REVOLIZER|ROTAHALER"),
    ("MDI", r"INHALER|\bMDI\b|EVOHALER"),
    ("NEB", r"\bRESP|\bNEB\b|NEBULI|TRANSPULE|MISTPULE|INHALATION"),
    ("INJ", r"\bINJ|\bVIALS?\b|\bAMPS?\b|AMPOULE|\bPFS\b|INFUSION|\bIV\b|\bPEN\b|CARTRIDGE|PREFILLED|\bPB\b|\bBAG\b|\bFFS\b"),
    ("TAB", r"\bTABS?\b|TABLET"),
    ("CAP", r"\bCAPS?\b|CAPSULE"),
    ("ORAL LIQUID", r"\bSYP\b|SYRUP|\bSYR\b|\bSUSP|\bLIQ|ORAL SOL|ELIXIR"),
    ("TOPICAL", r"CREAM|\bOINT|\bGEL\b|LOTION"),
    ("DROPS", r"\bDROPS?\b"),
    ("SPRAY", r"SPRAY"),
    ("POWDER", r"SACHET|\bSACH\b|POWDER|\bPWD\b"),
    ("SUPP", r"SUPPOS"),
    ("PATCH", r"PATCH"),
)
# Kinds of the same molecule that are not substitutes for the plain one. A brand is named
# where neither its description nor its Item Master minor group says the kind: LIPODOX is
# liposomal doxorubicin filed under plain DOXORUBICIN (Rs 9,000+ a vial against Rs 200).
KINDS = (
    ("MR", r"\b(?:SR|ER|XR|MR|CR|CD|XL|LA|OD|PR)\b|RETARD|PROLONGED|SUSTAINED|EXTENDED|CONTROLLED"),
    ("LIPO", r"LIPOSOM|PEGYLAT|\bLIPODOX\b"),
    ("NAB", r"PROTEIN.?BOUND|ALBUMIN.?BOUND|\bNAB\b"),
)
_KINDS = tuple((k, re.compile(rx)) for k, rx in KINDS)
_FORMS = tuple((k, re.compile(rx)) for k, rx in FORMS)


def _num(v: float) -> str:
    return f"{v:.6g}"


def strength_key(desc) -> str:
    """'ZAVIEDGE 2GM/0.5MG INJ VIAL' -> '0.5MG|2000MG'; '1 GM' and '1000MG' agree. The doses
    written in the description, plus its volumes when the dose alone does not fix the pack:
    no dose at all, only a % ('NS 0.9% 250ML' -> '0.9%|250ML'), or a dose per mL
    ('100IU/ML 3ML PEN' -> '100IU|3ML'). With no unit at all, the bare numbers ('DOLO 650
    TAB' -> '650'). '' when nothing is written."""
    s = K.norm_text(desc)
    dose, vol = [], []
    for n, unit in _AMOUNT.findall(s):
        if unit in _DOSE:
            u, f = _DOSE[unit]
            dose.append(_num(float(n) * f) + u)
        else:
            u, f = _VOLUME[unit]
            vol.append(_num(float(n) * f) + u)
    if not dose and not vol:
        return "|".join(sorted({_num(float(n)) for n in _BARE.findall(s)}))
    with_volume = not dose or all(d.endswith("%") for d in dose) or bool(_PER_ML.search(s))
    return "|".join(sorted(set(dose + (vol if with_volume else []))))


def form_key(desc) -> str:
    """'BUDECORT 200MCG ROTACAP' -> 'DPI': the first FORMS rule the description matches; ''
    when none does."""
    s = K.norm_text(desc)
    for k, rx in _FORMS:
        if rx.search(s):
            return k
    return ""


def kind_key(desc) -> str:
    """'TRAMADOL 100MG SR TAB' -> 'MR': the first KINDS rule the description matches; '' for
    the plain product."""
    s = K.norm_text(desc)
    for k, rx in _KINDS:
        if rx.search(s):
            return k
    return ""

