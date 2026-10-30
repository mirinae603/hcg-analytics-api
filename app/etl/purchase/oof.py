"""Out-of-formulary purchases, HCG's way: every OUT OF FORMULARY receipt, the formulary brand
HCG would have bought instead, and the margin missed.

Lines: the purchase scope (config.Scope) whose material the month's own Item Master flags OUT
OF FORMULARY. Values per line are HCG's worksheet formulas:

  actual     abb_mrp = Unit MRP / (100 + Tax %) x 100 (the MRP without GST)
             total_pp = Base Price x GR Qty; total_mrp = GR Qty x abb_mrp
             actual_margin = total_mrp - total_pp
  suggested  sug_abb = MRP / (100 + GST %) x 100, with the suggested brand's HCG Pharma
             selling price, MRP (incl GST) and GST %
             sug_total_pp = price x GR Qty; sug_total_mrp = sug_abb x GR Qty
             suggested_margin = sug_total_mrp - sug_total_pp
  loss       margin_loss = suggested_margin - actual_margin. HCG nets it: a line whose brand
             earns more than the suggestion lowers the loss (gain_loss + better_loss).

Which brand is suggested (suggestion_source), first that applies:
  1 hcg_sheet       HCG's worksheet of the same month (key period + plant + material; when
                    the key carries several suggestions, the one of the line's base price,
                    else the most common), with HCG's typed price / MRP / GST as typed. A line
                    HCG parked ('Need To check flaging later') is status 'parked'; a line HCG
                    left without a brand (or not on its sheet) stays 'none': with this source
                    every centre equals HCG's Summary to the paisa. Sources 2-5 are used only
                    for a month without HCG's worksheet.
  2 hcg_sheet_pair  HCG's suggestion for the same OOF material in its nearest worksheet month
                    (earlier months first, then later): HCG works from one fixed list.
  3 hcg_list        HCG's OOF master, as given (the master row's price, MRP and GST).
  4 formulary_rule  our rule ('Estimated'): a CASH&TPA brand the month's Item Master flags
                    FORMULARY, matched on molecule, strength, form and kind (RULE_LEVELS),
                    HCG's P1 of its block first.
  5 none            no alternative: shown, adds nothing to the margins or the loss.
Prices for sources 2-4 (rule PC): the master row's price for a hcg_list brand; else the
lowest master price of that brand when the master lists it; else its CASH&TPA FY price. A
suggested brand none of them prices is status 'unpriced' (adds nothing, like 'none').

The golden guard (abort_reasons): with HCG's worksheet of a month, every centre must equal HCG's
Summary and KR HCG's email within a rupee. A month HCG's email has figures for but whose
workbook was not read aborts as well, unless the build is told to publish it from sources 2-5.

Line-level frames stay in memory: the published table (OOF_COLUMNS) is aggregated to period
x plant x material x vendor x suggestion, and carries no GR, PO or invoice number.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Collection, Iterable, Mapping, Optional, Sequence

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary.discover import BuildError
from app.etl.purchase import config as C
from app.etl.purchase import molecule as Mo
from app.etl.purchase.masters import (PAIR_COLUMNS, PARKED_COLUMNS, SUMMARY_COLUMNS, HcgWorkbook,
                                      MainSheet, OofMaster, OofMasterRow)
from app.etl.spend.masters import PricedBrand

# --- Vocabulary of the published table ----------------------------------------------------
HCG_SHEET, HCG_SHEET_PAIR, HCG_LIST, FORMULARY_RULE, NONE = (
    "hcg_sheet", "hcg_sheet_pair", "hcg_list", "formulary_rule", "none")
SOURCES = (HCG_SHEET, HCG_SHEET_PAIR, HCG_LIST, FORMULARY_RULE, NONE)
COUNTED, PARKED, NO_ALTERNATIVE, UNPRICED = "counted", "parked", "no_alternative", "unpriced"
STATUSES = (COUNTED, PARKED, NO_ALTERNATIVE, UNPRICED)
# Where a suggestion's price, MRP and GST came from.
PRICE_HCG_SHEET, PRICE_LIST_ROW, PRICE_LIST_LOWEST, PRICE_CASH_TPA = (
    "hcg_sheet", "hcg_list_row", "hcg_list_lowest", "cash_tpa")
PRICE_SOURCES = (PRICE_HCG_SHEET, PRICE_LIST_ROW, PRICE_LIST_LOWEST, PRICE_CASH_TPA)

# Our rule's levels, tried in this order; the first with a candidate decides.
RULE_LEVELS = {
    "A": "same molecule, strength, dosage form and kind",
    "B": "same molecule, strength and kind; tablet for capsule (or oral liquid for drops)",
    "C": "same molecule, dose, form group and kind; another pack size",
    "D": "the OOF brand names no dose: same molecule and form group, pack within 4x",
    "F": "the molecule inside a combination (or the reverse); same strength, form group, kind",
}
FUZZY_MARK = "~"                 # appended to a level when the molecule matched by spelling
# The values each vocabulary column of oof_purchase may hold (missing aside): the API labels
# them, so a value outside these is a build error, not a blank label.
VOCABULARY = {"status": STATUSES, "suggestion_source": SOURCES, "price_source": PRICE_SOURCES,
              "rule_level": tuple(RULE_LEVELS) + tuple(k + FUZZY_MARK for k in RULE_LEVELS)}
MOLECULE_SIMILARITY = 0.8        # difflib ratio for a molecule key spelt differently
MOLECULE_ALTERNATIVES = 3
PACK_RATIO_MAX = 4.0             # level D: pack within 4x either way
NO_PACK_DISTANCE = 9.0           # tie-break distance when a pack size is not written

# HCG's emailed figures the build checks KR against (rounded rupees), and the guard.
EMAIL_FILE = "oof_email_figures.csv"
MAX_HCG_DIFF = 1.0               # rupees: a centre (or KR vs the email) further off aborts

# --- The published table: oof_purchase ----------------------------------------------------
GROUP_KEYS = ["period", "plant", "material", "vendor_code", "status", "suggestion_source",
              "rule_level", "suggested_material", "price_source", "sug_price", "sug_mrp", "sug_gst"]
ATTRIBUTES = ["category", "segment", "material_desc", "minor_group", "manufacturer",
              "vendor_name", "suggested_desc"]
SUMS = ["lines", "qty", "total_pp", "total_wo_tax", "total_mrp", "actual_margin",
        "sug_total_pp", "sug_total_mrp", "suggested_margin", "margin_loss", "gain_loss",
        "better_loss"]
TEXT_COLUMNS = ["period", "plant", "category", "segment", "material", "material_desc",
                "minor_group", "manufacturer", "vendor_code", "vendor_name", "status",
                "suggestion_source", "rule_level", "suggested_material", "suggested_desc",
                "price_source"]
PRICE_COLUMNS = ["sug_price", "sug_mrp", "sug_gst", "base_price_min", "base_price_max"]
OOF_COLUMNS = TEXT_COLUMNS + PRICE_COLUMNS + SUMS
OOF_SORT = ["period", "plant", "material", "vendor_code", "status", "suggestion_source",
            "suggested_material", "sug_price"]
# Per-line value columns (in memory only).
VALUE_COLUMNS = ["abb_mrp", "total_pp", "total_mrp", "actual_margin", "sug_total_pp",
                 "sug_total_mrp", "suggested_margin", "margin_loss"]
# HCG's Summary columns as our sums give them (C, D, E, H, I, J, L).
CENTRE_COLUMNS = ["used_pp", "used_mrp", "used_margin", "sug_pp", "sug_mrp", "sug_margin", "loss"]


# --------------------------------------------------------------------------- #
# Suggestions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Suggestion:
    """The brand suggested for one OOF line (or material), and how it was found."""
    source: str = NONE
    status: str = NO_ALTERNATIVE
    material: Optional[str] = None          # the suggested formulary brand
    desc: Optional[str] = None
    price: Optional[float] = None           # HCG Pharma selling price, excl GST
    mrp: Optional[float] = None             # incl GST
    gst: Optional[float] = None             # %
    price_source: Optional[str] = None
    level: Optional[str] = None             # formulary_rule only: RULE_LEVELS key (+ FUZZY_MARK)


NO_SUGGESTION = Suggestion()
PARKED_LINE = Suggestion(source=HCG_SHEET, status=PARKED)
SUGGESTION_COLUMNS = {"source": "suggestion_source", "status": "status",
                      "material": "suggested_material", "desc": "suggested_desc",
                      "price": "sug_price", "mrp": "sug_mrp", "gst": "sug_gst",
                      "price_source": "price_source", "level": "rule_level"}


@dataclass(frozen=True)
class Price:
    price: float
    mrp: float
    gst: float
    source: str


def _priced(source: str, material: str, desc: Optional[str], price: Optional[Price],
            level: Optional[str] = None) -> Suggestion:
    if price is None:
        return Suggestion(source=source, status=UNPRICED, material=material, desc=desc, level=level)
    return Suggestion(source=source, status=COUNTED, material=material, desc=desc,
                      price=price.price, mrp=price.mrp, gst=price.gst,
                      price_source=price.source, level=level)


class PriceBook:
    """Rule PC: the price, MRP and GST of a suggested brand from sources 2-4."""

    def __init__(self, master: OofMaster, cash_prices: Mapping[str, PricedBrand]):
        self._lowest: dict = {}
        for e in sorted(master.entries.values(), key=lambda e: (e.price, e.row)):
            self._lowest.setdefault(e.suggested, Price(e.price, e.mrp, e.gst, PRICE_LIST_LOWEST))
        self._cash = {m: Price(p.price, p.mrp, p.gst, PRICE_CASH_TPA) for m, p in cash_prices.items()}

    @staticmethod
    def master_row(e: OofMasterRow) -> Price:
        """A hcg_list brand: the master row's own price (one suggested brand carries two
        prices on different OOF rows)."""
        return Price(e.price, e.mrp, e.gst, PRICE_LIST_ROW)

    def for_brand(self, material: str) -> Optional[Price]:
        """The lowest master price of the brand when the master lists it, else its CASH&TPA FY
        price; None when neither has it."""
        return self._lowest.get(material) or self._cash.get(material)


# --- Source 1: HCG's worksheet of the same month -------------------------------------------
def _pair_suggestion(row) -> Suggestion:
    if pd.isna(row.suggested) or any(pd.isna(v) for v in (row.sug_price, row.sug_mrp, row.sug_gst)):
        return NO_SUGGESTION                         # HCG's own #N/A: HCG left it without a brand
    return Suggestion(source=HCG_SHEET, status=COUNTED, material=row.suggested,
                      desc=None if pd.isna(row.suggested_desc) else row.suggested_desc,
                      price=float(row.sug_price), mrp=float(row.sug_mrp), gst=float(row.sug_gst),
                      price_source=PRICE_HCG_SHEET)


def sheet_suggestions(lines: pd.DataFrame, wb: HcgWorkbook) -> list:
    """Source 1 for each line (plant, material, base_price): parked when HCG parked that
    plant x material x base price; else HCG's suggestion for the plant x material (when HCG
    gave it several, the one of the line's base price, else the one on most lines); else
    NO_SUGGESTION."""
    parked = set(zip(wb.parked["plant"], wb.parked["material"], wb.parked["base_price"]))
    by_key: dict = {}
    for row in wb.pairs.itertuples(index=False):
        by_key.setdefault((row.plant, row.material), []).append(row)
    memo: dict = {}
    out = []
    for key in zip(lines["plant"], lines["material"], lines["base_price"]):
        if key not in memo:
            memo[key] = _sheet_choice(key, by_key.get(key[:2], []), parked)
        out.append(memo[key])
    return out


def _most_common(pairs: list) -> Counter:
    lines = Counter()
    for p in pairs:
        lines[_pair_suggestion(p)] += p.lines
    return lines


def _sheet_choice(key: tuple, pairs: list, parked: set) -> Suggestion:
    if key in parked:
        return PARKED_LINE
    if not pairs:
        return NO_SUGGESTION
    options = _most_common(pairs)
    if len(options) > 1:
        options = _most_common([p for p in pairs if p.base_price == key[2]]) or options
    return max(options, key=options.get)          # a tie: the first in HCG's pair order


# --- Sources 2-5 --------------------------------------------------------------------------
@dataclass(frozen=True)
class SheetPair:
    """HCG's suggestion for one OOF material in one worksheet month."""
    period: str
    suggested: str
    desc: Optional[str]


def sheet_pair_book(wb: HcgWorkbook) -> dict:
    """{OOF material: SheetPair} of one HCG worksheet: the suggestion HCG gave the material
    on most lines that month (lines HCG could not price are left out)."""
    p = wb.pairs[wb.pairs["suggested"].notna()]
    g = (p.groupby(["material", "suggested"], sort=True)
         .agg(lines=("lines", "sum"), desc=("suggested_desc", "first")).reset_index()
         .sort_values(["material", "lines"], ascending=[True, False], kind="stable"))
    return {r.material: SheetPair(wb.period, r.suggested, None if pd.isna(r.desc) else r.desc)
            for r in g.drop_duplicates("material").itertuples(index=False)}


def pair_months(period: str, sheet_months: Iterable[str], earlier_only: bool = False) -> list:
    """The worksheet months source 2 reads for `period`, in order: earlier months nearest
    first, then (unless earlier_only) later months nearest first."""
    months = sorted(set(sheet_months) - {period})
    earlier = [m for m in reversed(months) if m < period]
    return earlier if earlier_only else earlier + [m for m in months if m > period]


@dataclass(frozen=True)
class PoolBrand:
    """A formulary brand our rule may suggest: a priced CASH&TPA row of a FORMULARY material."""
    material: str
    material_desc: Optional[str]
    product: Mo.Product
    block_id: int                   # the CASH&TPA block (interchangeable brands)
    position: int                   # order within the block, as in the sheet
    p_rank: int                     # 1 = HCG's P1 (highest 2026-27 SHARING in the block)


@dataclass(frozen=True)
class RuleMatch:
    brand: PoolBrand
    level: str


def _pack_ok(oof: Optional[float], cand: Optional[float]) -> bool:
    if not oof or cand is None:
        return True
    return 1 / PACK_RATIO_MAX <= cand / oof <= PACK_RATIO_MAX


def _nested(a: Optional[str], b: Optional[str]) -> bool:
    pa, pb = Mo.molecule_parts(a), Mo.molecule_parts(b)
    return a != b and bool(pa) and bool(pb) and (pa <= pb or pb <= pa)


def level_matches(level: str, o: Mo.Product, b: Mo.Product) -> bool:
    """Whether candidate b meets RULE_LEVELS[level] for the OOF product o."""
    same_mol = b.molecule == o.molecule
    if level == "A":
        return same_mol and b.strength == o.strength and b.form == o.form and b.kind == o.kind
    if level == "B":
        return (same_mol and b.strength == o.strength and b.form_group == o.form_group
                and b.kind == o.kind)
    if level == "C":
        return (o.dose != "" and same_mol and b.dose == o.dose and b.form_group == o.form_group
                and b.kind == o.kind)
    if level == "D":
        return (o.dose == "" and same_mol and _pack_ok(o.pack, b.pack)
                and (b.form_group == o.form_group or "" in (b.form_group, o.form_group)))
    if level == "F":
        return (_nested(o.molecule, b.molecule) and b.strength == o.strength
                and b.form_group == o.form_group and b.kind == o.kind)
    raise ValueError(f"unknown rule level {level!r}")


def _preference(o: Mo.Product, b: PoolBrand) -> tuple:
    """Same kind first, then HCG's P1, then most shared descriptor words, then the nearest
    pack size, then the sheet order."""
    p = b.product
    pack = abs(math.log(p.pack / o.pack)) if (p.pack and o.pack) else NO_PACK_DISTANCE
    return (p.kind != o.kind, b.p_rank, -len(o.words & p.words), pack, b.block_id, b.position)


class FormularyPool:
    """The brands our rule chooses from for one month, and the rule itself (match)."""

    def __init__(self, brands: Iterable[PoolBrand]):
        self.brands = tuple(brands)
        self._by_molecule: dict = {}
        for b in self.brands:
            self._by_molecule.setdefault(b.product.molecule, []).append(b)

    def __len__(self) -> int:
        return len(self.brands)

    @classmethod
    def from_main_sheet(cls, blocks: pd.DataFrame, prices: Mapping[str, PricedBrand],
                        formulary: set, minor_groups: Mapping[str, str]) -> "FormularyPool":
        """CASH&TPA brands (blocks in sheet order) with an FY price and MRP, FORMULARY in the
        month's Item Master and a molecule (minor_groups: material -> Minor Group)."""
        brands = []
        position = blocks.groupby("block_id").cumcount()
        for r, pos in zip(blocks.itertuples(index=False), position):
            if r.material not in prices or r.material not in formulary:
                continue
            product = Mo.describe(minor_groups.get(r.material), r.material_desc)
            if product.molecule is None:
                continue
            brands.append(PoolBrand(material=r.material, material_desc=r.material_desc,
                                    product=product, block_id=int(r.block_id), position=int(pos),
                                    p_rank=int(r.p_rank)))
        return cls(brands)

    def match(self, o: Mo.Product, levels: str = "".join(RULE_LEVELS)) -> Optional[RuleMatch]:
        """The brand for OOF product o: the first level with a candidate decides, the
        preference order picks within it. A molecule no pool brand has is tried under up to
        three keys spelt nearly the same (level marked FUZZY_MARK); level F, which compares
        the parts of a combination, tries the molecule itself first."""
        if not o.molecule:
            return None
        exact = o.molecule in self._by_molecule
        similar = [] if exact else Mo.close_molecules(o.molecule, self._by_molecule,
                                                      MOLECULE_ALTERNATIVES, MOLECULE_SIMILARITY)
        for level in levels:
            molecules = [o.molecule] if exact else ([o.molecule] if level == "F" else []) + similar
            for mol in molecules:
                oo = o if mol == o.molecule else replace(o, molecule=mol)
                pool = self.brands if level == "F" else self._by_molecule.get(mol, ())
                cands = [b for b in pool if level_matches(level, oo, b.product)]
                if cands:
                    best = min(cands, key=lambda b: _preference(oo, b))
                    return RuleMatch(best, level + ("" if mol == o.molecule else FUZZY_MARK))
        return None


@dataclass
class Chain:
    """Sources 2-5 for one month: HCG's pairs from other worksheet months (in order), HCG's
    master, our rule on the month's pool, and the prices (rule PC)."""
    pair_books: Sequence[dict]
    master: OofMaster
    pool: FormularyPool
    prices: PriceBook

    def suggest(self, material: str, product: Mo.Product) -> Suggestion:
        for book in self.pair_books:
            pair = book.get(material)
            if pair is not None:
                return _priced(HCG_SHEET_PAIR, pair.suggested, pair.desc,
                               self.prices.for_brand(pair.suggested))
        row = self.master.entries.get(material)
        if row is not None:
            return _priced(HCG_LIST, row.suggested, row.suggested_desc, PriceBook.master_row(row))
        m = self.pool.match(product)
        if m is not None:
            return _priced(FORMULARY_RULE, m.brand.material, m.brand.material_desc,
                           self.prices.for_brand(m.brand.material), m.level)
        return NO_SUGGESTION

    def for_lines(self, lines: pd.DataFrame, products: Mapping[str, Mo.Product]) -> list:
        memo: dict = {}
        out = []
        for m in lines["material"]:
            if m not in memo:
                memo[m] = self.suggest(m, products[m])
            out.append(memo[m])
        return out


# --------------------------------------------------------------------------- #
# Lines and values
# --------------------------------------------------------------------------- #
def out_of_formulary(lines: pd.DataFrame, item_master: pd.DataFrame) -> pd.DataFrame:
    """The scope's lines of one month whose material the month's Item Master (frame of
    masters.item_master_frame) flags OUT OF FORMULARY, with the master's description, minor
    group and manufacturer (the GRN's own where the master has none), category and segment."""
    im = item_master.set_index("material")
    cls = lines["material"].map(im["formulary_class"])
    out = lines[cls.eq(K.OUT_OF_FORMULARY)].reset_index(drop=True)
    for col in ("material_desc", "minor_group", "manufacturer"):
        out[col] = out["material"].map(im[col]).fillna(out[col])
    out["category"] = out["material"].map(K.category)
    out["segment"] = out["category"].map(K.SEGMENT_KEY)
    return out


def products(lines: pd.DataFrame) -> dict:
    """{material: Product} of the OOF lines (Item Master minor group and description)."""
    first = lines.drop_duplicates("material")
    return {m: Mo.describe(g, d) for m, g, d in
            zip(first["material"], first["minor_group"], first["material_desc"])}


def with_suggestions(lines: pd.DataFrame, suggestions: Sequence[Suggestion]) -> pd.DataFrame:
    """The lines with one suggestion each (SUGGESTION_COLUMNS) and HCG's per-line values
    (VALUE_COLUMNS). Margins and the loss are zero on a line that is not counted."""
    if len(suggestions) != len(lines):
        raise ValueError("one suggestion per line")
    sug = pd.DataFrame([{SUGGESTION_COLUMNS[k]: getattr(s, k) for k in SUGGESTION_COLUMNS}
                        for s in suggestions], columns=list(SUGGESTION_COLUMNS.values()),
                       index=lines.index)
    for c in ("sug_price", "sug_mrp", "sug_gst"):
        sug[c] = sug[c].astype("float64")
    out = pd.concat([lines, sug], axis=1)
    qty = out["gr_qty"]
    counted = out["status"].eq(COUNTED)
    out["abb_mrp"] = out["unit_mrp"] / (100 + out["tax_pct"]) * 100
    out["total_pp"] = out["base_price"] * qty
    out["total_mrp"] = qty * out["abb_mrp"]
    sug_abb = out["sug_mrp"] / (100 + out["sug_gst"]) * 100
    out["sug_total_pp"] = (out["sug_price"] * qty).where(counted, 0.0)
    out["sug_total_mrp"] = (qty * sug_abb).where(counted, 0.0)
    out["actual_margin"] = (out["total_mrp"] - out["total_pp"]).where(counted, 0.0)
    out["suggested_margin"] = out["sug_total_mrp"] - out["sug_total_pp"]
    out["margin_loss"] = out["suggested_margin"] - out["actual_margin"]
    return out


def aggregate(lines: pd.DataFrame) -> pd.DataFrame:
    """Valued lines -> oof_purchase rows (OOF_COLUMNS): one row per period x plant x material
    x vendor x suggestion (status, source, level, brand, price source, price, MRP, GST)."""
    df = lines.assign(lines=1, qty=lines["gr_qty"],
                      gain_loss=lines["margin_loss"].clip(lower=0.0),
                      better_loss=lines["margin_loss"].clip(upper=0.0))
    g = df.groupby(GROUP_KEYS, dropna=False, sort=False)
    out = g.agg(**{c: (c, "first") for c in ATTRIBUTES},
                base_price_min=("base_price", "min"), base_price_max=("base_price", "max"),
                **{c: (c, "sum") for c in SUMS}).reset_index()
    out = out[OOF_COLUMNS]
    out[TEXT_COLUMNS] = out[TEXT_COLUMNS].astype("str")
    out[PRICE_COLUMNS + SUMS] = out[PRICE_COLUMNS + SUMS].astype("float64")
    out["lines"] = out["lines"].astype("int64")
    return out.sort_values(OOF_SORT, kind="stable", ignore_index=True)


def vocabulary_problems(table: pd.DataFrame) -> list:
    """oof_purchase rows -> one problem per vocabulary column holding a value VOCABULARY does
    not list."""
    out = []
    for column, allowed in VOCABULARY.items():
        bad = sorted(set(table[column].dropna().astype(str)) - set(allowed))
        if bad:
            out.append(f"oof_purchase.{column} holds {', '.join(bad)}, outside its vocabulary "
                       f"({', '.join(allowed)})")
    return out


def centre_figures(table: pd.DataFrame) -> pd.DataFrame:
    """oof_purchase rows -> HCG's Summary columns per period x plant (CENTRE_COLUMNS): used
    and suggested PP / MRP over the counted rows, the two margins and the loss."""
    c = table[table["status"].eq(COUNTED)]
    g = c.groupby(["period", "plant"]).agg(used_pp=("total_pp", "sum"), used_mrp=("total_mrp", "sum"),
                                           used_margin=("actual_margin", "sum"),
                                           sug_pp=("sug_total_pp", "sum"),
                                           sug_mrp=("sug_total_mrp", "sum"),
                                           sug_margin=("suggested_margin", "sum"),
                                           loss=("margin_loss", "sum"))
    return g.reset_index()[["period", "plant"] + CENTRE_COLUMNS]


# --------------------------------------------------------------------------- #
# HCG's emailed KR figures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmailFigure:
    """One hospital-month row of HCG's spend email (rounded rupees)."""
    period: str
    plant: str
    actual_margin: float
    suggested_margin: float
    margin_loss: float
    source: str


def load_email_figures(ref_dir=None) -> dict:
    """oof_email_figures.csv (period, plant, actual_margin, suggested_margin, margin_loss,
    source) -> {(period, plant): EmailFigure}. No file: no email check (the Summary check
    still covers every centre)."""
    path = C.reference_dir(ref_dir) / EMAIL_FILE
    if not path.exists():
        return {}
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            where = f"{path.name} line {n}"
            period = (row.get("period") or "").strip()
            plant = K.norm_text(row.get("plant"))
            if not K.is_period(period) or not plant:
                raise BuildError(f"{where}: needs a period (YYYY-MM) and a plant")
            try:
                v = {k: float(row[k]) for k in ("actual_margin", "suggested_margin", "margin_loss")}
            except (KeyError, TypeError, ValueError):
                raise BuildError(f"{where}: actual_margin, suggested_margin and margin_loss must "
                                 f"be numbers") from None
            if (period, plant) in out:
                raise BuildError(f"{where}: {plant} {period} listed twice")
            out[(period, plant)] = EmailFigure(period, plant, source=(row.get("source") or "").strip(), **v)
    return out


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def _pct(num, den):
    return None if not den else round(100.0 * num / den, 2)


def _counts(lines: pd.DataFrame, by: str, values=("total_pp",)) -> dict:
    g = lines.groupby(by, dropna=False)
    out = {}
    for k, part in g:
        key = k[0] if isinstance(k, tuple) else k
        out["" if pd.isna(key) else str(key)] = {"lines": int(len(part)),
                                                 **{v: _r(part[v].sum()) for v in values}}
    return dict(sorted(out.items()))


def totals(lines: pd.DataFrame) -> dict:
    """Headline sums of valued lines (or oof_purchase rows): HCG's net loss and its parts."""
    loss = lines["margin_loss"]
    sug = float(lines["suggested_margin"].sum())
    return {"actual_margin": _r(lines["actual_margin"].sum()), "suggested_margin": _r(sug),
            "margin_loss": _r(loss.sum()), "gain_loss": _r(loss.clip(lower=0).sum()),
            "better_loss": _r(loss.clip(upper=0).sum()), "loss_pct": _pct(float(loss.sum()), sug)}


def compare_centres(ours: pd.DataFrame, hcg: pd.DataFrame) -> dict:
    """Our loss per centre (CENTRE_COLUMNS of one month) beside HCG's Summary."""
    o = ours.set_index("plant")["loss"]
    h = hcg.set_index("plant")
    names = h["centre_name"]
    rows = []
    for plant in sorted(set(o.index) | set(h.index)):
        a, b = float(o.get(plant, 0.0)), float(h["loss"].get(plant, 0.0))
        rows.append({"plant": plant, "centre_name": names.get(plant), "ours": _r(a), "hcg": _r(b),
                     "diff": _r(a - b), "in_hcg": plant in h.index, "in_ours": plant in o.index})
    ours_total, hcg_total = float(o.sum()), float(h["loss"].sum())
    diffs = [abs(r["diff"]) for r in rows]
    return {"loss_ours": _r(ours_total), "loss_hcg": _r(hcg_total), "diff": _r(ours_total - hcg_total),
            "diff_pct": _pct(ours_total - hcg_total, abs(hcg_total)),
            "sum_abs_centre_diff": _r(sum(diffs)), "max_centre_diff": _r(max(diffs, default=0.0)),
            "centres_within_5pct": sum(1 for r in rows
                                       if r["hcg"] and abs(r["diff"]) <= 0.05 * abs(r["hcg"])),
            "centres": rows}


def _kr(ours: pd.DataFrame, hcg: Optional[pd.DataFrame], email: Optional[EmailFigure]) -> dict:
    o = ours[ours["plant"] == C.KR_PLANT]
    mine = {"actual_margin": _r(o["used_margin"].sum()), "suggested_margin": _r(o["sug_margin"].sum()),
            "margin_loss": _r(o["loss"].sum()),
            "loss_pct": _pct(float(o["loss"].sum()), float(o["sug_margin"].sum()))}
    out = {"plant": C.KR_PLANT, "ours": mine, "hcg_summary": None, "email": None, "diff_vs_email": None}
    if hcg is not None and (hcg["plant"] == C.KR_PLANT).any():
        h = hcg[hcg["plant"] == C.KR_PLANT].iloc[0]
        out["hcg_summary"] = {"actual_margin": _r(h["used_margin"]), "suggested_margin": _r(h["sug_margin"]),
                              "margin_loss": _r(h["loss"])}
    if email is not None:
        e = {"actual_margin": email.actual_margin, "suggested_margin": email.suggested_margin,
             "margin_loss": email.margin_loss}
        out["email"] = {**e, "source": email.source}
        out["diff_vs_email"] = {k: _r(mine[k] - v) for k, v in e.items()}
        out["diff_vs_email_pct"] = _pct(mine["margin_loss"] - email.margin_loss, abs(email.margin_loss))
    return out


def _brands_reproduced(sim: pd.DataFrame, sheet: pd.DataFrame) -> dict:
    """How often a simulated suggestion is HCG's own brand, over the lines HCG priced."""
    hcg = sheet["status"].eq(COUNTED)
    same = hcg & sim["suggested_material"].eq(sheet["suggested_material"]).fillna(False)
    return {"hcg_lines": int(hcg.sum()), "same_brand_lines": int(same.sum()),
            "same_brand_pct": _pct(int(same.sum()), int(hcg.sum()))}


def pair_check(lines: pd.DataFrame, wb: HcgWorkbook) -> dict:
    """Our OOF lines and qty per HCG pair key (plant x material x base price) beside HCG's."""
    key = ["plant", "material", "base_price"]
    ours = lines.groupby(key).agg(lines=("material", "size"), qty=("gr_qty", "sum"))
    hcg = pd.concat([wb.pairs[key + ["lines", "qty"]], wb.parked[key + ["lines", "qty"]]])
    hcg = hcg.groupby(key).agg(lines=("lines", "sum"), qty=("qty", "sum"))
    j = hcg.join(ours, how="left", rsuffix="_ours").fillna(0.0)
    differ = (j["lines"] != j["lines_ours"]) | ((j["qty"] - j["qty_ours"]).abs() > 1e-6)
    only_ours = ours.index.difference(hcg.index)
    return {"hcg_keys": int(len(hcg)), "keys_lines_or_qty_differ": int(differ.sum()),
            "hcg_lines_not_found": int((j["lines"] - j["lines_ours"]).clip(lower=0).sum()),
            "our_keys_not_on_hcg_sheet": int(len(only_ours)),
            "our_lines_not_on_hcg_sheet": int(ours.loc[only_ours, "lines"].sum())}


def simulate(lines: pd.DataFrame, sheet_lines: pd.DataFrame, chain: Chain, prods: dict,
             hcg: pd.DataFrame, email: Optional[EmailFigure], months_used: list) -> dict:
    """'A month without HCG's worksheet': sources 2-5 with only the worksheets of earlier
    months, on every OOF line of the month, beside HCG's Summary."""
    sim = with_suggestions(lines, chain.for_lines(lines, prods))
    centres = centre_figures(sim)
    cmp = compare_centres(centres, hcg)
    return {"rule": "sources 2-5 (HCG's pairs from earlier worksheets, HCG's master, our rule), "
                    "every OOF line counted as if HCG had no worksheet this month",
            "pair_months": months_used,
            "by_source": _counts(sim, "suggestion_source", ("total_pp", "margin_loss")),
            "by_status": _counts(sim, "status"),
            "rule_levels": _counts(sim[sim["rule_level"].notna()], "rule_level", ()),
            **totals(sim), **{k: v for k, v in cmp.items() if k != "centres"},
            "brands": _brands_reproduced(sim, sheet_lines), "centres": cmp["centres"],
            "kr": _kr(centres, hcg, email)}


def month_report(period: str, valued: pd.DataFrame, *, item_master_file: Optional[str],
                 wb: Optional[HcgWorkbook], email: Optional[EmailFigure], pool_size: int,
                 pair_months_used: list) -> dict:
    """The OOF section of one month's validation report (counts and rupees only)."""
    centres = centre_figures(valued)
    hcg = wb.summary if wb is not None else None
    rep = {
        "item_master": {"file": item_master_file, "out_of_formulary_lines": int(len(valued)),
                        "total_wo_tax": _r(valued["total_wo_tax"].sum()),
                        "total_pp": _r(valued["total_pp"].sum())},
        "suggestions_from": "HCG's worksheet of the month (source 1)" if wb is not None else
                            "sources 2-5: HCG's pairs from other worksheets, HCG's master, our rule",
        "pair_months": pair_months_used,
        "formulary_pool_brands": pool_size,
        "by_status": _counts(valued, "status"),
        "by_source": _counts(valued, "suggestion_source", ("total_pp", "margin_loss")),
        "rule_levels": _counts(valued[valued["rule_level"].notna()], "rule_level", ()),
        "price_sources": _counts(valued[valued["price_source"].notna()], "price_source", ()),
        **totals(valued),
        "hcg": None,
        "kr": _kr(centres, hcg, email),
        "warnings": [],
    }
    if wb is not None:
        rep["hcg"] = {**wb.report(), **compare_centres(centres, wb.summary),
                      "pair_check": pair_check(valued, wb)}
    if email is not None and wb is None:
        rep["warnings"].append(f"{period}: HCG's email has KR figures but HCG's worksheet is "
                               f"missing; our figures come from sources 2-5")
    return rep


def abort_reasons(period: str, rep: dict, *, sheet_waived: bool = False) -> list:
    """The golden guard: with HCG's worksheet of the month, every centre must equal HCG's
    Summary and KR HCG's email, each within MAX_HCG_DIFF rupees. Without it, a month HCG's
    email has figures for aborts too: HCG made a worksheet for that month, so its workbook is
    missing (moved, renamed or unreadable) and sources 2-5 would quietly stand in for HCG's
    own figures. sheet_waived publishes such a month from sources 2-5 all the same."""
    out = []
    hcg = rep.get("hcg")
    if not hcg:
        if (rep.get("kr") or {}).get("email") is not None and not sheet_waived:
            out.append(f"{period}: HCG's email has KR figures for this month but HCG's out-of-"
                       f"formulary workbook was not read; restore it, or rebuild with "
                       f"--without-hcg-sheet {period} to publish the month from sources 2-5")
        return out
    for c in hcg["centres"]:
        if abs(c["diff"] or 0.0) > MAX_HCG_DIFF:
            out.append(f"{period}: centre {c['plant']} loss {c['ours']:,.2f} vs HCG's Summary "
                       f"{c['hcg']:,.2f}")
    diff = (rep.get("kr") or {}).get("diff_vs_email") or {}
    for k, v in diff.items():
        if v is not None and abs(v) > MAX_HCG_DIFF:
            out.append(f"{period}: KR {k} {rep['kr']['ours'][k]:,.2f} vs HCG's email "
                       f"{rep['kr']['email'][k]:,.2f}")
    return out


# --------------------------------------------------------------------------- #
# The tables of the build
# --------------------------------------------------------------------------- #
@dataclass
class OofBuild:
    """What build_tables returns (build.Built's fields; the build-wide section is
    report_section, taken on the published table)."""
    tables: dict = field(default_factory=dict)      # table name -> DataFrame (built months)
    months: dict = field(default_factory=dict)      # period -> this build's month report
    problems: list = field(default_factory=list)    # abort reasons: nothing is published
    replaces: dict = field(default_factory=dict)    # table name -> the months its rows replace


def _minor_groups(item_masters: Mapping[str, pd.DataFrame], period: str) -> dict:
    """material -> Minor Group from the month's Item Master, then the other months' (nearest
    first) for a material the month's master leaves blank."""
    order = sorted(item_masters, key=lambda p: (p != period, abs(_month_index(p) - _month_index(period)), p))
    out: dict = {}
    for p in order:
        im = item_masters[p]
        for m, g in zip(im["material"], im["minor_group"]):
            if m not in out and isinstance(g, str) and g.strip():
                out[m] = g
    return out


def _month_index(period: str) -> int:
    y, m = map(int, period.split("-"))
    return 12 * y + m


def _check_years(periods: list, master: OofMaster, main: MainSheet) -> None:
    for what, fy, covers in (("OOF master", master.fy, master.covers),
                             ("CASH&TPA list", main.cash.fy, main.cash.covers)):
        outside = [p for p in periods if not covers(p)]
        if outside:
            raise BuildError(f"the {what} is for FY {fy[0]}-{str(fy[1])[2:]}; it does not cover "
                             f"{', '.join(outside)} (HCG to send that year's list)")


def hcg_tables(hcg: Mapping[str, HcgWorkbook], periods: Iterable[str]) -> dict:
    """HCG's own figures for the months built whose workbook was read: Summary, pairs and
    parked lines."""
    wbs = [hcg[p] for p in sorted(set(hcg) & set(periods))]

    def cat(frames, columns):
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)

    return {C.HCG_SUMMARY: cat([w.summary for w in wbs], SUMMARY_COLUMNS),
            C.HCG_PAIRS: cat([w.pairs for w in wbs], PAIR_COLUMNS),
            C.HCG_PARKED: cat([w.parked for w in wbs], PARKED_COLUMNS)}


def build_tables(*, lines: pd.DataFrame, periods: Sequence[str], item_masters: Mapping[str, pd.DataFrame],
                 item_master_files: Mapping[str, str], hcg: Mapping[str, HcgWorkbook],
                 master: OofMaster, main: MainSheet, email: Mapping[tuple, EmailFigure],
                 without_sheet: Collection[str] = (), log: Callable = print) -> OofBuild:
    """oof_purchase (+ HCG's Summary, pairs and parked lines) for every built month that has
    its own Item Master, with each month's report and the golden guard's abort reasons.
    without_sheet: months to publish from sources 2-5 although HCG's email has figures for
    them (their workbook is missing). HCG's own tables replace only the months whose workbook
    was read: a month built without it keeps the HCG figures published before."""
    oof_periods = [p for p in sorted(periods) if p in item_masters]
    out = OofBuild()
    if not oof_periods:
        return out
    _check_years(oof_periods, master, main)
    books = {p: sheet_pair_book(wb) for p, wb in hcg.items()}
    prices = PriceBook(master, main.prices)
    parts = []
    for p in oof_periods:
        oof = out_of_formulary(lines[lines["period"] == p], item_masters[p])
        im = item_masters[p]
        formulary = set(im.loc[im["formulary_flag"] == K.FORMULARY, "material"])
        pool = FormularyPool.from_main_sheet(main.blocks, main.prices, formulary,
                                             _minor_groups(item_masters, p))
        prods = products(oof)
        wb = hcg.get(p)
        live = pair_months(p, hcg)
        chain = Chain([books[m] for m in live], master, pool, prices)
        sug = sheet_suggestions(oof, wb) if wb is not None else chain.for_lines(oof, prods)
        valued = with_suggestions(oof, sug)
        parts.append(aggregate(valued))
        rep = month_report(p, valued, item_master_file=item_master_files.get(p), wb=wb,
                           email=email.get((p, C.KR_PLANT)), pool_size=len(pool),
                           pair_months_used=[] if wb is not None else live)
        if wb is not None:
            rep["no_pair"] = _no_pair(valued, oof, chain, prods, live)
            earlier = pair_months(p, hcg, earlier_only=True)
            sim_chain = Chain([books[m] for m in earlier], master, pool, prices)
            rep["simulation"] = simulate(oof, valued, sim_chain, prods, wb.summary,
                                         email.get((p, C.KR_PLANT)), earlier)
        out.months[p] = rep
        out.problems += abort_reasons(p, rep, sheet_waived=p in without_sheet)
        log(_summary_line(p, rep))
    read = [p for p in oof_periods if p in hcg]
    out.tables = {C.OOF: pd.concat(parts, ignore_index=True), **hcg_tables(hcg, read)}
    out.replaces = {C.OOF: oof_periods, **{name: read for name in C.HCG_TABLES}}
    out.problems += vocabulary_problems(out.tables[C.OOF])
    return out


def _no_pair(valued: pd.DataFrame, oof: pd.DataFrame, chain: Chain, prods: dict, live: list) -> dict:
    """The lines of an HCG month that HCG left without a brand, and what sources 2-4 (with
    the other worksheets) would give them: reported, never counted."""
    none = valued["suggestion_source"].eq(NONE)
    sub = oof[none]
    out = {"lines": int(none.sum()), "total_pp": _r(valued.loc[none, "total_pp"].sum())}
    if len(sub):
        v = with_suggestions(sub, chain.for_lines(sub, prods))
        priced = v[v["status"].eq(COUNTED)]
        out["sources_2_4_could_price"] = {"pair_months": live, "lines": int(len(priced)),
                                          "total_pp": _r(priced["total_pp"].sum()),
                                          "margin_loss": _r(priced["margin_loss"].sum()),
                                          "by_source": _counts(priced, "suggestion_source", ("total_pp",))}
    return out


def _summary_line(period: str, rep: dict) -> str:
    kr = rep["kr"]["ours"]
    s = (f"[purchase] {period}: OOF {rep['item_master']['out_of_formulary_lines']:,} lines, loss "
         f"{rep['margin_loss']:,.2f} ({rep['loss_pct']}% of {rep['suggested_margin']:,.2f}); KR "
         f"{kr['actual_margin']:,.2f} / {kr['suggested_margin']:,.2f} / {kr['margin_loss']:,.2f}")
    if rep.get("hcg"):
        h = rep["hcg"]
        sim = rep["simulation"]
        s += (f"\n[purchase] {period}: HCG's Summary {h['loss_hcg']:,.2f} (ours - HCG "
              f"{h['diff']:,.2f}, worst centre {h['max_centre_diff']:,.2f}); simulation without "
              f"the sheet {sim['margin_loss']:,.2f} ({sim['diff_pct']}% vs HCG)")
    return s


def report_section(master: OofMaster, main: MainSheet, hcg: Mapping[str, HcgWorkbook],
                   table: Optional[pd.DataFrame], email: Mapping) -> dict:
    """The build-wide OOF section of purchase_validation.json: the months the published
    oof_purchase holds (this build's and the ones kept), the method and the masters."""
    periods = sorted(table["period"].astype(str).unique()) if table is not None else []
    if not periods:
        return {"months": [], "note": "no built month has its own Item Master"}
    fm = master.frame()
    return {
        "months": periods,
        "rules": {
            "lines": "purchase scope lines whose material the month's own Item Master flags OUT OF "
                     "FORMULARY (not the GRN's Formulary column)",
            "actual": "abb_mrp = Unit MRP / (100 + Tax %) x 100; total_pp = Base Price x GR Qty; "
                      "total_mrp = GR Qty x abb_mrp; actual margin = total_mrp - total_pp",
            "suggested": "sug_abb = MRP / (100 + GST %) x 100; sug_total_pp = price x GR Qty; "
                         "sug_total_mrp = sug_abb x GR Qty; suggested margin = sug_total_mrp - "
                         "sug_total_pp",
            "loss": "margin_loss = suggested margin - actual margin, netted (HCG); gain_loss = "
                    "the lines' positive losses, better_loss = the negative ones",
            "sources": {HCG_SHEET: "HCG's worksheet of the same month, HCG's typed price/MRP/GST",
                        HCG_SHEET_PAIR: "HCG's pair for the material from its nearest worksheet "
                                        "month (earlier first), priced by rule PC",
                        HCG_LIST: "HCG's OOF master, the row's own price/MRP/GST",
                        FORMULARY_RULE: "our rule (Estimated): " + "; ".join(
                            f"{k} {v}" for k, v in RULE_LEVELS.items()) + "; within a level: same "
                            "kind, HCG's P1, most shared descriptor words, nearest pack, sheet order",
                        NONE: "no alternative: shown, adds nothing to the margins or the loss"},
            "prices_pc": "master row price for a hcg_list brand; else the lowest master price of "
                         "the brand; else its CASH&TPA FY price, MRP and GST",
            "statuses": {COUNTED: "in the margins and the loss", PARKED: "HCG doubts the flag "
                         "('Need To check flaging later'): shown, not counted",
                         NO_ALTERNATIVE: "no suggested brand", UNPRICED: "a suggested brand no "
                         "price list prices: shown, not counted"},
            "guard": f"with HCG's worksheet of the month, every centre within Rs {MAX_HCG_DIFF:g} of "
                     f"HCG's Summary and KR within Rs {MAX_HCG_DIFF:g} of HCG's email, else nothing "
                     f"is published; a month HCG's email has figures for but whose workbook was not "
                     f"read is not published either, unless the build is run with --without-hcg-sheet",
        },
        "oof_master": {"file": master.name, "sheet": master.sheet,
                       "financial_year": f"{master.fy[0]}-{str(master.fy[1])[2:]}" if master.fy else None,
                       "rows": master.rows, "materials": len(master.entries),
                       "suggested_brands": int(fm["suggested"].nunique()), "skipped": master.skipped},
        "main_sheet": {"file": main.path.name, "cash_fy": f"{main.cash.fy[0]}-{str(main.cash.fy[1])[2:]}",
                       "priced_brands": len(main.prices), "block_rows": int(len(main.blocks))},
        "hcg_workbooks": {p: {"file": wb.name, "ws_sheet": wb.ws_sheet, "warnings": wb.warnings}
                          for p, wb in hcg.items()},
        "email": {"file": EMAIL_FILE, "rows": [f"{p} {pl}" for p, pl in sorted(email)]},
    }
