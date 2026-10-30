"""Generic vs branded purchases: which purchases are an originator company's brand, how each
side splits by formulary class, and what the originator brands cost above the cheapest
generic of the same product.

The definition (decision D12; shown as an 'Estimated definition' until HCG confirms it):
  branded   a product of an ORIGINATOR (innovator) company: the GRN 'Manufacturer' starts with
            the words of a pattern in app/data/reference/originator_manufacturers.csv
            (pattern, company, reason) — Roche, MSD, AstraZeneca, BMS, Pfizer, ...
  generic   every other product: Indian brands, biosimilars, Attis own label, Jan Aushadhi,
            MNC generic and hospital makers (Fresenius Kabi, Baxter, B Braun, ...)
  override  app/data/reference/generic_overrides.csv (material, class, reason) decides first:
            originator brands sold in India under an Indian licensee's name (VECTIBIX by Dr
            Reddy's, ...)
class_source says which applied (CLASS_SOURCES): override, originator_list, default (no
pattern matched: generic).

Lines: the purchase scope of every GRN month, all formulary classes, as prices.priced_lines
gives them (Item Master names, product key, unit price and its status). Per period x plant x
material: value (Total Amount without tax), qty, lines, the class, and the month's own Item
Master class — FORMULARY with its tier (TIER_GROUPS), NON FORMULARY, OUT OF FORMULARY, NOT IN
MASTER — for the months that have an Item Master (D7; empty for the others).

Cost impact (extra_vs_generic): a branded line with a comparable unit price (prices.OK) whose
product (molecule_key) was also bought as a generic that month, at any centre, costs
(unit price - the cheapest generic's price) x GR Qty more, when positive. The cheapest
generic's price is the USUAL cheapest of the product's generic lines that month (prices.OK
lines, every centre; prices.usual_cheapest: the lowest unit price at which at least
prices.USUAL_SHARE of the generic quantity was bought), so 15 vials of one generic bought once
at a fraction of every other generic's price do not become the benchmark for a whole month of
branded purchases; generic_material names the generic bought at that price. compared_value is
the value of the branded lines priced against it.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary.discover import BuildError
from app.etl.purchase import config as C
from app.etl.purchase import prices as P

# --- Vocabulary -------------------------------------------------------------------------------
BRANDED, GENERIC = "branded", "generic"
CLASSES = (BRANDED, GENERIC)
OVERRIDE, ORIGINATOR_LIST, DEFAULT = "override", "originator_list", "default"
CLASS_SOURCES = {
    OVERRIDE: "generic_overrides.csv names the material",
    ORIGINATOR_LIST: "the GRN manufacturer is an originator company (originator_manufacturers.csv)",
    DEFAULT: "no originator pattern matches the manufacturer: generic",
}
ORIGINATOR_FILE = "originator_manufacturers.csv"
OVERRIDES_FILE = "generic_overrides.csv"
DEFINITION = ("Branded = a product of an originator (innovator) company; generic = every other "
              "product (Indian brands, biosimilars, Attis own label, Jan Aushadhi, MNC generic and "
              "hospital makers). Estimated definition until HCG confirms it.")
# HCG's tier letters on a FORMULARY item (Item Master Category) in four groups.
PREFERRED, SCHEME_ONLY, RESTRICTED, OTHER_TIER = "preferred", "scheme_only", "restricted", "other"
TIER_GROUPS = {"P": PREFERRED, "P-SCH": PREFERRED, "P-OPD": PREFERRED, "SCH": SCHEME_ONLY,
               "SCH-OPD": SCHEME_ONLY, "R": RESTRICTED, "R-OPD": RESTRICTED, "OPD": OTHER_TIER,
               "CONSUMABLE": OTHER_TIER}
TOP_EXTRA = 10                   # report: branded materials listed per month by extra paid

# --- The published table: purchase_generic ---------------------------------------------------
GROUP_KEYS = ["period", "plant", "material", "manufacturer"]
ATTRIBUTES = ["category", "segment", "material_desc", "minor_group", "class", "class_source",
              "originator_company"] + P.PRODUCT_COLUMNS + [
              "formulary_flag", "formulary_class", "tier", "tier_group", "generic_material",
              "generic_desc", "generic_price"]
TEXT_COLUMNS = ["period", "plant", "category", "segment", "material", "material_desc",
                "minor_group", "manufacturer", "class", "class_source", "originator_company"
                ] + P.PRODUCT_COLUMNS + ["formulary_flag", "formulary_class", "tier", "tier_group",
                                         "generic_material", "generic_desc"]
SUMS = ["lines", "qty", "value", "compared_value", "extra_vs_generic"]
GENERIC_COLUMNS = TEXT_COLUMNS + ["generic_price"] + SUMS
GENERIC_SORT = ["period", "material", "plant"]


# --------------------------------------------------------------------------- #
# The definition (reference files)
# --------------------------------------------------------------------------- #
_WORDS = re.compile(r"[A-Z0-9]+")


def name_words(name) -> tuple:
    """A company name as words: upper case, apostrophes dropped, split on anything else that
    is not a letter or digit. "DR.REDDY'S" -> ('DR', 'REDDYS'); 'J & J' and 'J&J' -> ('J', 'J');
    () for a blank name ('Blank', NaN)."""
    if K.is_blank(name):
        return ()
    return tuple(_WORDS.findall(K.norm_text(name).replace("'", "")))


@dataclass(frozen=True)
class Originator:
    """One row of originator_manufacturers.csv: a manufacturer whose name starts with
    `words` is the originator `company`."""
    pattern: str
    words: tuple
    company: str
    reason: str


@dataclass(frozen=True)
class Override:
    """One row of generic_overrides.csv."""
    material: str
    cls: str
    reason: str


@dataclass(frozen=True)
class Classification:
    cls: str
    source: str
    company: Optional[str] = None


@dataclass(frozen=True)
class Definition:
    """The originator list and the material overrides: which purchases are branded."""
    originators: tuple = ()
    overrides: Mapping[str, Override] = field(default_factory=dict)

    def originator(self, manufacturer) -> Optional[Originator]:
        """The first pattern (file order) whose words begin the manufacturer's name."""
        words = name_words(manufacturer)
        for o in self.originators:
            if words[:len(o.words)] == o.words:
                return o
        return None

    def classify(self, material: str, manufacturer) -> Classification:
        """A material's class: its override, else branded when its manufacturer is an
        originator, else generic."""
        o = self.originator(manufacturer)
        company = o.company if o else None
        if material in self.overrides:
            return Classification(self.overrides[material].cls, OVERRIDE, company)
        if o is not None:
            return Classification(BRANDED, ORIGINATOR_LIST, company)
        return Classification(GENERIC, DEFAULT)


def _rows(path, required: tuple):
    if not path.exists():
        raise BuildError(f"generic vs branded reference file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise BuildError(f"{path.name}: needs the columns {', '.join(required)}")
        for n, row in enumerate(reader, start=2):
            yield f"{path.name} line {n}", row


def load_originators(ref_dir=None) -> tuple:
    """originator_manufacturers.csv (pattern, company, reason) -> Originators in file order.
    A pattern needs a word and a company; a pattern listed twice is an error."""
    out, seen = [], {}
    for where, row in _rows(C.reference_dir(ref_dir) / ORIGINATOR_FILE, ("pattern", "company", "reason")):
        pattern, company = K.norm_text(row["pattern"]), (row["company"] or "").strip()
        words = name_words(pattern)
        if not words or not company:
            raise BuildError(f"{where}: needs a pattern and a company")
        if words in seen:
            raise BuildError(f"{where}: pattern {pattern} is already on line {seen[words]}")
        seen[words] = where.rsplit(" ", 1)[-1]
        out.append(Originator(pattern, words, company, (row["reason"] or "").strip()))
    return tuple(out)


def load_overrides(ref_dir=None) -> dict:
    """generic_overrides.csv (material, class, reason) -> {material: Override}. The class is
    branded or generic; a material listed twice is an error."""
    out: dict = {}
    for where, row in _rows(C.reference_dir(ref_dir) / OVERRIDES_FILE, ("material", "class", "reason")):
        material, cls = K.norm_code(row["material"]), (row["class"] or "").strip().lower()
        if not material or cls not in CLASSES:
            raise BuildError(f"{where}: needs a material and a class ({' or '.join(CLASSES)})")
        if material in out:
            raise BuildError(f"{where}: material {material} listed twice")
        out[material] = Override(material, cls, (row["reason"] or "").strip())
    return out


def load_definition(ref_dir=None) -> Definition:
    return Definition(originators=load_originators(ref_dir), overrides=load_overrides(ref_dir))


# --------------------------------------------------------------------------- #
# Lines
# --------------------------------------------------------------------------- #
def with_classes(priced: pd.DataFrame, definition: Definition) -> pd.DataFrame:
    """priced_lines with class, class_source and originator_company (one call per material x
    manufacturer)."""
    pairs = priced[["material", "manufacturer"]].drop_duplicates()
    got = [definition.classify(m, f) for m, f in zip(pairs["material"], pairs["manufacturer"])]
    pairs = pairs.assign(**{"class": [c.cls for c in got], "class_source": [c.source for c in got],
                            "originator_company": [c.company for c in got]})
    return priced.merge(pairs, on=["material", "manufacturer"], how="left", validate="many_to_one")


def with_formulary(lines: pd.DataFrame, item_masters: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The lines with the formulary_flag, formulary_class, tier and tier_group of their own
    month's Item Master (NOT IN MASTER for a material it lacks); empty for a month without one."""
    out = lines.copy()
    for col in ("formulary_flag", "formulary_class", "tier"):
        out[col] = pd.Series(pd.NA, index=out.index, dtype="str")
    for period, im in item_masters.items():
        rows = out["period"] == period
        if not rows.any():
            continue
        m = im.drop_duplicates("material").set_index("material")
        mat = out.loc[rows, "material"]
        out.loc[rows, "formulary_class"] = mat.map(m["formulary_class"]).fillna(K.NOT_IN_MASTER)
        out.loc[rows, "formulary_flag"] = mat.map(m["formulary_flag"]).fillna(K.NOT_IN_MASTER)
        out.loc[rows, "tier"] = mat.map(m["tier"])
    formulary = out["formulary_flag"].eq(K.FORMULARY)
    out["tier_group"] = out["tier"].map(TIER_GROUPS).where(formulary)
    out.loc[formulary & out["tier_group"].isna(), "tier_group"] = OTHER_TIER
    return out


def cheapest_generics(lines: pd.DataFrame) -> pd.DataFrame:
    """Per period x molecule_key: the usual cheapest unit price of the product's comparable
    generic lines (every centre; prices.usual_cheapest over GR Qty) and the generic material
    bought at it -> generic_price, generic_material, generic_desc."""
    ok = lines[lines["price_status"].eq(P.OK) & lines["molecule_key"].notna()
               & lines["class"].eq(GENERIC)]
    ok = ok.sort_values(["material"], kind="stable")          # equal prices: the lowest code
    best = P.usual_cheapest(ok, ["period", "molecule_key"], "unit_price", qty="gr_qty",
                            carry=("material", "material_desc")).reset_index()
    return best.rename(columns={"unit_price": "generic_price", "material": "generic_material",
                                "material_desc": "generic_desc"})[
        ["period", "molecule_key", "generic_material", "generic_desc", "generic_price"]]


def with_cost_impact(lines: pd.DataFrame) -> pd.DataFrame:
    """The lines with the cheapest generic of their product and month, compared_value and
    extra_vs_generic (branded lines with a comparable price only; 0 on every other line)."""
    out = lines.merge(cheapest_generics(lines), on=["period", "molecule_key"], how="left",
                      validate="many_to_one")
    compared = (out["class"].eq(BRANDED) & out["price_status"].eq(P.OK)
                & out["generic_price"].notna())
    out["compared_value"] = out["total_wo_tax"].where(compared, 0.0)
    out["extra_vs_generic"] = ((out["unit_price"] - out["generic_price"]).clip(lower=0.0)
                               * out["gr_qty"]).where(compared, 0.0)
    return out


def classified_lines(priced: pd.DataFrame, definition: Definition,
                     item_masters: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """prices.priced_lines -> + class, Item Master class and the cost impact. In memory only."""
    return with_cost_impact(with_formulary(with_classes(priced, definition), item_masters))


def aggregate(lines: pd.DataFrame) -> pd.DataFrame:
    """classified_lines -> purchase_generic rows (GENERIC_COLUMNS): one row per period x plant
    x material (x manufacturer: one per material and month in HCG's GRN)."""
    g = lines.assign(lines=1, qty=lines["gr_qty"], value=lines["total_wo_tax"]).groupby(
        GROUP_KEYS, dropna=False, sort=False)
    out = g.agg(**{c: (c, "first") for c in ATTRIBUTES}, **{c: (c, "sum") for c in SUMS}).reset_index()
    out = out[GENERIC_COLUMNS]
    out[TEXT_COLUMNS] = out[TEXT_COLUMNS].astype("str")
    out[["generic_price"] + SUMS[1:]] = out[["generic_price"] + SUMS[1:]].astype("float64")
    out["lines"] = out["lines"].astype("int64")
    return out.sort_values(GENERIC_SORT, kind="stable", ignore_index=True)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _r(v, nd=2) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _split(table: pd.DataFrame, by: str) -> dict:
    """{class: {value of each `by` value}} over the rows that have one."""
    t = table[table[by].notna()]
    out: dict = {}
    for (cls, key), v in t.groupby(["class", by])["value"].sum().items():
        out.setdefault(cls, {})[key] = _r(v)
    return {cls: dict(sorted(out[cls].items())) for cls in CLASSES if cls in out}


def month_report(table: pd.DataFrame) -> dict:
    """The generic section of one month's validation report (counts and rupees only)."""
    total = float(table["value"].sum())
    by = table.groupby("class").agg(lines=("lines", "sum"), value=("value", "sum"),
                                    materials=("material", "nunique"))
    src = table.groupby("class_source").agg(lines=("lines", "sum"), value=("value", "sum"))
    branded = table[table["class"].eq(BRANDED)]
    extra = (branded[branded["extra_vs_generic"] > 0]
             .groupby(["material", "material_desc", "manufacturer", "generic_desc"], dropna=False)
             ["extra_vs_generic"].sum().sort_values(ascending=False).head(TOP_EXTRA))
    over = (table[table["class_source"].eq(OVERRIDE)]
            .groupby(["material", "material_desc", "class"])["value"].sum())
    has_im = table["formulary_flag"].notna().any()
    return {
        "lines": int(table["lines"].sum()), "value": _r(total),
        "by_class": {c: {"lines": int(by.loc[c, "lines"]), "value": _r(by.loc[c, "value"]),
                         "materials": int(by.loc[c, "materials"]),
                         "value_pct": _r(100 * by.loc[c, "value"] / total) if total else None}
                     for c in CLASSES if c in by.index},
        "by_source": {s: {"lines": int(src.loc[s, "lines"]), "value": _r(src.loc[s, "value"])}
                      for s in CLASS_SOURCES if s in src.index},
        "overrides": [{"material": m, "material_desc": d, "class": c, "value": _r(v)}
                      for (m, d, c), v in over.items()],
        "formulary": _split(table, "formulary_flag") if has_im else None,
        "tiers": _split(table, "tier_group") if has_im else None,
        "cost_impact": {
            "extra_vs_generic": _r(table["extra_vs_generic"].sum()),
            "compared_value": _r(table["compared_value"].sum()),
            "branded_value_not_compared": _r(branded["value"].sum() - branded["compared_value"].sum()),
            "top": [{"material": m, "material_desc": d, "manufacturer": f, "cheapest_generic": g,
                     "extra_vs_generic": _r(v)} for (m, d, f, g), v in extra.items()],
        },
    }


def summary_line(months: dict) -> str:
    """One log line: the months built and the latest month's headline."""
    if not months:
        return "[purchase] generic: no month built"
    p = max(months)
    rep = months[p]
    by = rep["by_class"]
    side = "; ".join(f"{c} {by[c]['value']:,.2f} ({by[c]['value_pct']}%)" for c in CLASSES if c in by)
    return (f"[purchase] generic: {len(months)} months; {p}: {side}; originator brands above the "
            f"cheapest generic {rep['cost_impact']['extra_vs_generic']:,.2f}")


def report_section(definition: Definition, table: pd.DataFrame) -> dict:
    """The build-wide generic section of purchase_validation.json: the definition, its files
    and the manufacturers it counted as originators in the months built."""
    matched = (table[table["class_source"].eq(ORIGINATOR_LIST)]
               .groupby("manufacturer")["originator_company"].first())
    return {
        "definition": DEFINITION, "estimated": True,
        "files": {"originators": ORIGINATOR_FILE, "overrides": OVERRIDES_FILE},
        "originator_patterns": len(definition.originators),
        "originator_companies": len({o.company for o in definition.originators}),
        "overrides": [{"material": o.material, "class": o.cls, "reason": o.reason}
                      for o in definition.overrides.values()],
        "manufacturers_counted_as_originators": {k: v for k, v in sorted(matched.items())},
        "rules": {
            "class_source": CLASS_SOURCES,
            "formulary": "the month's own Item Master: FORMULARY (tier groups "
                         + ", ".join(f"{k} {v}" for k, v in TIER_GROUPS.items())
                         + "), NON FORMULARY, OUT OF FORMULARY, NOT IN MASTER; empty for a month "
                           "without an Item Master",
            "cost_impact": f"a branded line with a comparable unit price (prices ok) whose product "
                           f"(molecule_key) was bought as a generic that month: (unit price - the "
                           f"usual cheapest generic price, the lowest at which at least "
                           f"{P.USUAL_SHARE:.0%} of the generic quantity was bought) x GR Qty, when "
                           f"positive",
        },
    }


# --------------------------------------------------------------------------- #
# The table of the build
# --------------------------------------------------------------------------- #
def build_table(priced: pd.DataFrame, definition: Definition,
                item_masters: Mapping[str, pd.DataFrame]) -> tuple:
    """purchase_generic for the priced lines' months -> (table, {period: month report})."""
    table = aggregate(classified_lines(priced, definition, item_masters))
    months = {p: month_report(table[table["period"] == p]) for p in sorted(table["period"].unique())}
    return table, months
