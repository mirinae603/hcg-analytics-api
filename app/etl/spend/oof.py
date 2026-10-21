"""Out-of-formulary margin (Task 3c): the formulary brand that stands in for an
out-of-formulary (OOF) brand, found from HCG's own masters.

HCG's spend email puts an OUT OF FORMULARY row beside CASH(TPA) (actual margin, proposed
margin, loss) but has never sent the list of which formulary brand replaces which OOF brand.
Until it does, the build derives that list:

  same molecule   the Item Master's 'Minor Group' of the OOF brand equals that of a brand the
                  same month's Item Master flags FORMULARY
  same strength   the strengths written in both descriptions ('80MG', '2GM/0.5MG', '0.9%',
                  '4000 UNITS'), in one unit (1GM = 1000MG); volumes count only when the dose
                  alone does not fix the pack; a bare number ('DOLO 650 TAB') when nothing else
                  is written. A brand with no strength at all is never matched
  same form       injection, tablet, capsule, dry-powder / metered-dose inhaler, nebuliser, ...
                  read from the description
  same kind       modified release (SR, ER, XR, ...), liposomal / pegylated and protein-bound
                  products only match their own kind
  priced          the formulary brand has an FY price, MRP and GST % in the Main Sheet's
                  CASH&TPA sheet (the prices HCG's cash list uses)
  alike in price  its MRP is within OOF_PRICE_BAND (10x) of the OOF brand's own MRP per unit
                  on the month's bills: a far dearer or cheaper "match" is a different product
                  or pack (liposomal doxorubicin is ~15x plain doxorubicin), not a substitute
  best margin     of several such brands, the one with the highest unit margin (MRP excl GST
                  - HCG Pharma price), as HCG's cash list points to the higher-margin brand;
                  equal margins -> the lowest material code

Values, for cash and TPA patients (who pay MRP), in HCG's cash method:
  actual margin    = bill MRP / (1 + GST%) - bill cost   (an OOF brand has no FY list price,
                     so its own bill stands in; GST is the matched brand's, the same molecule,
                     strength and form)
  proposed margin  = qty x (suggested MRP / (1 + GST%) - suggested HCG Pharma price)
  margin loss      = proposed - actual; % = loss / proposed

An OOF brand with no match is still published (matched = False) with its bill values only,
so the coverage gap is visible; it adds nothing to the margins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.spend import config as C

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
# Kinds of the same molecule that are not substitutes for the plain one.
KINDS = (
    ("MR", r"\b(?:SR|ER|XR|MR|CR|CD|XL|LA|OD|PR)\b|RETARD|PROLONGED|SUSTAINED|EXTENDED|CONTROLLED"),
    ("LIPO", r"LIPOSOM|PEGYLAT"),
    ("NAB", r"PROTEIN.?BOUND|ALBUMIN.?BOUND|\bNAB\b"),
)
_KINDS = tuple((k, re.compile(rx)) for k, rx in KINDS)
_FORMS = tuple((k, re.compile(rx)) for k, rx in FORMS)

COLUMNS = ["material", "molecule", "strength", "form", "matched", "candidates", "suggested_material",
           "suggested_desc", "suggested_category", "suggested_price", "suggested_mrp", "suggested_gst"]


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
    s = K.norm_text(desc)
    for k, rx in _FORMS:
        if rx.search(s):
            return k
    return ""


def kind_key(desc) -> str:
    s = K.norm_text(desc)
    for k, rx in _KINDS:
        if rx.search(s):
            return k
    return ""


def molecule_key(minor_group) -> str:
    return K.norm_text(minor_group)


@dataclass(frozen=True)
class PricedBrand:
    """One brand row of the Main Sheet's CASH&TPA sheet with its FY prices."""
    material: str
    material_desc: str | None
    price: float                  # HCG Pharma selling price, excl GST
    mrp: float                    # MRP incl GST
    gst: float                    # %
    category: str | None          # the sheet's FY category (P, R, P-SCH, ...)

    @property
    def unit_margin(self) -> float:
        return self.mrp / (1 + self.gst / 100) - self.price


def is_formulary(cls) -> bool:
    return str(cls or "").startswith(K.FORMULARY)


def match(item_master, priced: dict, unit_mrp: dict = None, band: float = None) -> pd.DataFrame:
    """One row per OUT OF FORMULARY material of the month's Item Master: its molecule,
    strength and form, and the formulary brand chosen for it (COLUMNS).

    item_master: {material: Item(formulary_class, material_desc, minor_group, ...)}
    priced:      {material: PricedBrand} from the Main Sheet's CASH&TPA sheet
    unit_mrp:    {material: the OOF brand's MRP per unit on the month's bills}; a formulary
                 brand whose MRP is more than `band` times dearer or cheaper is not a match.
                 A brand with no bill price is matched on the other rules alone.
    """
    band = C.OOF_PRICE_BAND if band is None else band
    unit_mrp = unit_mrp or {}

    def key(item, fallback_desc=None):
        desc = item.material_desc or fallback_desc
        return molecule_key(item.minor_group), strength_key(desc), form_key(desc), kind_key(desc)

    pool: dict = {}
    for code, b in priced.items():
        it = item_master.get(code)
        if it is None or not is_formulary(it.formulary_class) or not molecule_key(it.minor_group):
            continue
        pool.setdefault(key(it, b.material_desc), []).append(b)

    rows = []
    for code, it in item_master.items():
        if it.formulary_class != K.OUT_OF_FORMULARY:
            continue
        k = key(it)
        mol, st, fm, _ = k
        cands = pool.get(k, []) if mol and st else []
        u = unit_mrp.get(code)
        if u and u > 0:
            cands = [b for b in cands if u / band <= b.mrp <= u * band]
        best = min(cands, key=lambda b: (-b.unit_margin, b.material)) if cands else None
        rows.append({"material": code, "molecule": mol or None, "strength": st or None, "form": fm or None,
                     "matched": best is not None, "candidates": len(cands),
                     "suggested_material": best.material if best else None,
                     "suggested_desc": best.material_desc if best else None,
                     "suggested_category": best.category if best else None,
                     "suggested_price": best.price if best else None,
                     "suggested_mrp": best.mrp if best else None,
                     "suggested_gst": best.gst if best else None})
    return pd.DataFrame(rows, columns=COLUMNS)


def values(df: pd.DataFrame) -> pd.DataFrame:
    """The margins of OOF rows (qty, actual_cost = bill cost, bill_mrp = bill MRP incl GST,
    and the match). Unmatched rows get no margins (NaN). Not rounded: totals are summed first."""
    df = df.copy()
    q = df["qty"].astype(float)
    gst = df["suggested_gst"].astype(float)
    df["used_gst"] = gst
    df["used_mrp_value"] = df["bill_mrp"].astype(float) / (1 + gst / 100)
    df["used_cost_value"] = df["actual_cost"].astype(float).where(df["matched"].astype(bool))
    df["suggested_mrp_value"] = q * df["suggested_mrp"].astype(float) / (1 + gst / 100)
    df["suggested_cost_value"] = q * df["suggested_price"].astype(float)
    df["actual_margin"] = df["used_mrp_value"] - df["used_cost_value"]
    df["potential_margin"] = df["suggested_mrp_value"] - df["suggested_cost_value"]
    df["margin_loss"] = df["potential_margin"] - df["actual_margin"]
    df["cash_qty"] = q
    return df
