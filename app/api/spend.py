"""Spend Analysis — HCG's Spend Margin Loss report: Attis (Task 2), scheme brands and cash &
TPA margin loss (Task 3).

Served from the small aggregated tables `python -m app.etl.spend` publishes into
settings.SPEND_DATA_DIR (line-level sales never reach the repo or this process):

    GET /spend/meta                 -> months, hospitals, filter options, the scheme list used
    GET /spend/attis/overview       -> headline, trend, hospital ranking (and the all-hospitals
                                       figure beside it, `network`), category and IP/OP
                                       split, top brands, Attis consumption summary
    GET /spend/attis/items          -> {data,total} per brand used; with Doctor=, one doctor's
    GET /spend/attis/doctors        -> {data,total} per treating doctor; with Material=, the
                                       doctors behind one brand
    GET /spend/attis/consumption    -> {data,total} Attis products billed, every payer
    GET /spend/scheme/overview      -> the same for EVERY scheme suggestion, plus the Attis and
                                       J&J & Covidien -> Healthium slices
    GET /spend/scheme/items         -> {data,total} per brand (Group=healthium|attis narrows)
    GET /spend/scheme/doctors       -> {data,total} per doctor (Group=, Material= as above)
    GET /spend/cash/overview        -> cash & TPA margin loss: headline, trend, ranking, splits,
                                       top brands, the bifurcation allowance, and HCG's 'CASH(TPA)
                                       & OUT OF FORMULARY' line (`with_oof`)
    GET /spend/cash/items           -> {data,total} per brand used; with Doctor=, one doctor's
    GET /spend/cash/doctors         -> {data,total} per doctor; with Material=, one brand's
    GET /spend/oof/overview         -> out-of-formulary margin loss on PURCHASES: headline with
                                       HCG's own figure beside it, trend, hospital ranking,
                                       onco / non-onco, top switches, vendors, where each
                                       suggestion came from, brands with no alternative
    GET /spend/oof/items            -> {data,total} per OOF brand x suggested brand; with
                                       Vendor=, one vendor's
    GET /spend/oof/vendors          -> {data,total} per vendor (goods supplier)
    GET /spend/brands               -> the brands the page's Brand filter offers, each with its
                                       switch in every section
    GET /spend/validation           -> the build's validation report (admin only)

HCG's rule: for a scheme (GOV) patient, a brand on HCG's scheme list should be replaced by the
cheaper brand the list suggests. Used value = scheme quantity x the used brand's FY list price;
suggested value = the same quantity x the suggested brand's list price; saving missed = used -
suggested; % = saving missed / used. The Attis views keep the rows whose suggested brand is an
Attis brand; the scheme views are the same functions over every row (attis_only=False).

Cash & TPA (CASH and CASH(TPA) patients pay MRP, so the loss is margin): unit margin = MRP
excl GST - HCG Pharma price, FY list, for the used and the suggested brand; actual margin =
counted qty x used margin; potential = counted qty x suggested margin; margin loss = potential
- actual; % = loss / potential (HCG's denominator — not the scheme %'s). Counted qty is the
cash/TPA qty except for a bifurcated brand, where the build already took off HCG's allowance.

Every figure is a ratio of SUMS taken after filtering — never an average of percentages — and
None when its denominator is 0.

Every list endpoint also returns `sums` (Contract A): its money and quantity columns added up
over every filtered row, not just the page, so a table can print a Total row under them. The lists
take the shared table filters too (formulary._page -> app.core.table_query: f.<field>= facets such
as FY class or molecule, literal search; facets=1 lists them and echoes the page filters as `scope`).

Out of formulary is HCG's PURCHASE method ('Out of Formulary Spend Analysis'), served from the
tables `python -m app.etl.purchase` publishes into settings.PURCHASE_DATA_DIR: oof_purchase
(every goods receipt in HCG's purchase scope of a drug the month's own Item Master flags OUT OF
FORMULARY, set against the formulary brand HCG would have bought) and oof_hcg_summary (HCG's own
per-centre Summary, shown beside ours). HCG's formulas, per GRN line: margin earned = qty x (MRP
excl GST - Base Price); margin possible = qty x the suggested brand's (MRP excl GST - HCG Pharma
price); margin missed = possible - earned, NET over the lines as HCG adds it (a brand earning
more than its suggestion lowers it: gain_loss + better_loss); % = missed / possible. Lines HCG
parked and lines with no suggested brand are shown and add nothing. Purchases have no payer,
patient type or doctor: Payer and Type are accepted and reported as not applied, and the
vendor takes the doctor's place. The cash overview adds the two margins together the way HCG's
email table does (`with_oof`).

Every figure is worked out from the published tables.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api import formulary as F
from app.api import purchase_insights as PI
from app.core import data_access as da
from app.core import table_query as tq
from app.core.config import settings
from app.core.deps import require_admin
from app.etl.formulary import classify as K
from app.etl.purchase import config as PC
from app.etl.purchase import oof as PO
from app.etl.purchase import prices as PP

router = APIRouter()

SWITCH, ATTIS_USE = "scheme_switch", "attis_consumption"
CASH_SWITCH, CASH_ALLOWANCE = "cash_switch", "cash_allowance"
TABLES = (SWITCH, ATTIS_USE, CASH_SWITCH, CASH_ALLOWANCE)
VALIDATION_FILE = "spend_validation.json"
# The purchase build's out-of-formulary tables, in settings.PURCHASE_DATA_DIR.
PURCHASE_TABLES = (PC.OOF, PC.HCG_SUMMARY)
NOT_BUILT = "Spend data has not been built yet — run python -m app.etl.spend"
NOT_BUILT_OOF = "The out-of-formulary purchase data has not been built yet — run python -m app.etl.purchase"

CATEGORIES = {"onco": "Onco Drugs", "non-onco": "Other Drugs", "consumables": "Consumables"}
CATEGORY_LABELS = {"onco": "Onco", "non-onco": "Non-Onco", "consumables": "Consumables"}
_CATEGORY_IN = {"ONCO": "onco", "NON-ONCO": "non-onco", "NON ONCO": "non-onco",
                "NON_ONCO": "non-onco", "NONONCO": "non-onco", "CONSUMABLES": "consumables",
                "CONSUMABLE": "consumables"}
# What each Segment keeps: its one category.
SEGMENT_CATEGORIES = {k: (v,) for k, v in CATEGORIES.items()}
TOP_N = 10                          # HCG's 'Top 10 Materials list contribution in margin loss'
TYPES = ("IP", "OP")
PAYERS = ("CASH", "CASH(TPA)", "GOV")
CASH_PAYERS = ("CASH", "CASH(TPA)")
VALUES = ["used_value", "suggested_value", "qty", "lines", "actual_cost"]
CASH_VALUES = ["actual_margin", "potential_margin", "margin_loss", "used_mrp_value", "suggested_mrp_value",
               "qty", "cash_qty", "lines", "actual_cost"]
# Scheme slices: HCG's 'J&J & Covidien vs Healthium' line is the SCHEME rows of these groups.
HEALTHIUM_GROUPS = ("SUTURES", "ENDO")
# HCG's S6 Summary shows the Healthium line as two rows, Sutures and Endo.
HEALTHIUM_PARTS = {"sutures": ("SUTURES", "Sutures"), "endo": ("ENDO", "Endo")}
GROUPS = {"ATTIS": "attis", "HEALTHIUM": "healthium"}
GROUP_LABELS = {"attis": "Attis brands", "healthium": "J&J & Covidien → Healthium"}
_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# ── Tables ────────────────────────────────────────────────────────────────────
# Read at CALL time from settings.SPEND_DATA_DIR (and PURCHASE_DATA_DIR for out of formulary)
# and keyed on each file's mtime/size, like the formulary tables: a rebuild or a test's own
# fixtures are picked up without a restart.
def _data_dir() -> Path:
    return Path(settings.SPEND_DATA_DIR)


def _purchase_dir() -> Path:
    return Path(settings.PURCHASE_DATA_DIR)


@lru_cache(maxsize=8)
def _read(path: str, stamp: tuple) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("attis", "bifurcated"):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df


def _table(name: str) -> pd.DataFrame:
    p = _data_dir() / f"{name}.parquet"
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return _read(str(p), stamp)


def _purchase_table(name: str) -> Optional[pd.DataFrame]:
    """One of the purchase build's tables; None until it has been built."""
    p = _purchase_dir() / f"{name}.parquet"
    stamp = F._stamp(p)
    return None if stamp is None else _read(str(p), stamp)


def _has_oof() -> bool:
    return F._stamp(_purchase_dir() / f"{PC.OOF}.parquet") is not None


def _version() -> tuple:
    d, p = _data_dir(), _purchase_dir()
    return ((str(d), str(p)) + tuple(F._stamp(d / f"{t}.parquet") for t in TABLES) + (F._stamp(d / VALIDATION_FILE),)
            + tuple(F._stamp(p / f"{t}.parquet") for t in PURCHASE_TABLES) + (F._stamp(p / PC.VALIDATION_FILE),))


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _validation() -> Optional[dict]:
    return _read_json(_data_dir() / VALIDATION_FILE)


_RESULT_CACHES: list = []


def _cache(fn):
    cached = lru_cache(maxsize=64)(fn)
    _RESULT_CACHES.append(cached)
    return cached


def clear_caches() -> None:
    _read.cache_clear()
    for c in _RESULT_CACHES:
        c.cache_clear()


# ── Arguments ─────────────────────────────────────────────────────────────────
class Scope(NamedTuple):
    plant: Optional[str]          # SAP plant code; None => all hospitals
    months: tuple                 # periods 'YYYY-MM'
    ptype: Optional[str]          # 'IP' / 'OP'; None => both
    category: Optional[str]       # 'onco' / 'non-onco' / 'consumables'; None => all
    material: Optional[str] = None  # one brand used (Material=); None => every brand


@_cache
def _periods_at(version: tuple) -> list:
    return sorted(set(_table(SWITCH)["period"].astype(str)) | set(_table(ATTIS_USE)["period"].astype(str)))


def _periods() -> list:
    return _periods_at(_version())


def _months_arg(month, periods: list, data: str = "spend") -> tuple:
    s = F._str(month)
    if s is None:
        return tuple(periods[-1:])          # latest month, like HCG's monthly report
    if s.upper() == "ALL":
        return tuple(periods)
    out = []
    for m in (x.strip() for x in s.split(",")):
        if not m:
            continue
        if not _PERIOD_RE.match(m):
            F._bad(f"Invalid Month '{m}'. Use YYYY-MM, a comma list of them, or ALL")
        if m not in periods:
            F._bad(f"No {data} data for {m}. Available months: {', '.join(periods)}")
        out.append(m)
    if not out:
        F._bad("Month is empty. Use YYYY-MM, a comma list of them, or ALL")
    return tuple(sorted(set(out)))


def _category_arg(segment, category) -> Optional[str]:
    """Segment=onco|non-onco|consumables, or the app-wide Category= (Onco Drugs / ...)."""
    seg = F._choice(segment, _CATEGORY_IN, "Segment")
    cat = da.resolve_category(F._str(category))
    if cat:
        by_cat = {v: k for k, v in CATEGORIES.items()}.get(cat)
        if by_cat is None:
            F._bad(f"Invalid Category '{category}'. Use Onco Drugs, Other Drugs or Consumables")
        if seg and CATEGORIES[by_cat] not in SEGMENT_CATEGORIES[seg]:
            F._bad(f"Segment '{segment}' and Category '{category}' disagree")
        seg = by_cat
    return seg


def _scope(Plant, Month, Type, Segment, Category, Material=None) -> Scope:
    periods = _periods()                    # 503 first: nothing below means anything without data
    return Scope(plant=F._plant_arg(Plant), months=_months_arg(Month, periods),
                 ptype=F._choice(Type, {t: t for t in TYPES}, "Type"),
                 category=_category_arg(Segment, Category), material=F._code(Material))


def _oof_scope(Plant, Month, Type, Segment, Category, Payer, Material=None) -> tuple:
    """(Scope, not_applied) for the out-of-formulary views: hospital, the purchase build's
    months, material type and brand. Purchases carry no patient type and no payer, so Type and
    Payer are checked like everywhere else and then named in `not_applied` instead of cutting."""
    periods = _oof_periods()                # 503 first
    ptype = F._choice(Type, {t: t for t in TYPES}, "Type")
    payer = F._choice(Payer, {p: p for p in PAYERS}, "Payer")
    sc = Scope(plant=F._plant_arg(Plant), months=_months_arg(Month, periods, "out-of-formulary purchase"),
               ptype=None, category=_oof_category_arg(Segment, Category), material=F._code(Material))
    return sc, tuple(name for name, given in (("Type", ptype), ("Payer", payer)) if given)


def _oof_category_arg(segment, category) -> Optional[str]:
    """_category_arg, limited to the material types the purchases hold: HCG's purchase scope
    (app/data/reference/purchase_scope_exclusions.csv) keeps the drug series only, so
    consumables are a 400 here, as on the Prices and Generic vs branded views."""
    seg = _category_arg(segment, category)
    present = [k for k, c in CATEGORIES.items() if c in set(_oof()["category"].astype(str))]
    if seg and seg not in present:
        F._bad(f"{CATEGORY_LABELS[seg]} are not in the out-of-formulary purchases — HCG's purchase scope covers "
               f"{' and '.join(CATEGORY_LABELS[k] for k in present) or 'no material type'} only")
    return seg


# ── Filtering + arithmetic ────────────────────────────────────────────────────
def _cut(df: pd.DataFrame, sc: Scope, *, plant=True, months=True, ptype=True,
         category=True) -> pd.DataFrame:
    # The brand filter (Material=) always applies: every view of a brand, its trend and its
    # hospital ranking included, is that brand's alone.
    m = pd.Series(True, index=df.index)
    if sc.material:
        m &= df["material"].astype(str) == sc.material
    if plant and sc.plant:
        m &= df["plant"].astype(str) == sc.plant
    if months:
        m &= df["period"].astype(str).isin(sc.months)
    if ptype and sc.ptype:
        m &= df["patient_type"].astype(str) == sc.ptype
    if category and sc.category:
        m &= df["category"].astype(str).isin(SEGMENT_CATEGORIES[sc.category])
    return df[m]


def _healthium(df: pd.DataFrame) -> pd.DataFrame:
    if "scheme_group" not in df.columns:            # tables built before Task 3
        return df.iloc[0:0]
    g = df["scheme_group"].fillna("").astype(str).str.strip().str.upper()
    return df[g.isin(HEALTHIUM_GROUPS)]


def _switches(attis_only: bool, group: Optional[str] = None) -> pd.DataFrame:
    sw = _table(SWITCH)
    if attis_only or group == "attis":
        sw = sw[sw["attis"]]
    if group == "healthium":
        sw = _healthium(sw)
    return sw


def _sums(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` ([] = one row): summed values, saving missed and saving %."""
    d = df[VALUES].astype(float)
    if keys:
        for k in keys:
            d[k] = df[k].astype(str)
        r = d.groupby(keys, sort=True).sum().reset_index()
    else:
        r = d.sum().to_frame().T
    r["saving_missed"] = r["used_value"] - r["suggested_value"]
    r["saving_pct"] = r["saving_missed"] / r["used_value"].where(r["used_value"] != 0) * 100
    return r


def _block(r) -> dict:
    return {"used_value": F._money(r["used_value"]), "suggested_value": F._money(r["suggested_value"]),
            "saving_missed": F._money(r["saving_missed"]), "saving_pct": F._pct(r["saving_pct"]),
            "qty": round(float(r["qty"]), 3) + 0.0, "lines": int(r["lines"]),
            "bill_cost": F._money(r["actual_cost"])}


EMPTY = {"used_value": 0.0, "suggested_value": 0.0, "saving_missed": 0.0, "saving_pct": None,
         "qty": 0.0, "lines": 0, "bill_cost": 0.0}


@_cache
def _hospitals_at(version: tuple) -> dict:
    """SAP plant -> {his_code, his_name} as the latest month's sales name it."""
    parts = [_table(t)[["period", "plant", "hospital", "hospital_name"]] for t in (SWITCH, ATTIS_USE)]
    s = pd.concat(parts, ignore_index=True).sort_values("period")
    last = s.drop_duplicates("plant", keep="last")
    return {str(r["plant"]): {"his_code": str(r["hospital"]), "his_name": str(r["hospital_name"])}
            for r in last.to_dict("records")}



def _hospitals() -> dict:
    return _hospitals_at(_version())


def _plant_ids() -> dict:
    """SAP plant -> {his_code, his_name}: the spend tables' (latest sales month), else the
    hospital map's. The out-of-formulary views use it too, and do not need the spend build."""
    hm = da.hospital_map()
    out = {} if hm.empty else {
        str(r["sap_plant"]): {"his_code": str(r["his_code"]), "his_name": str(r["his_location_name"])}
        for r in hm.drop_duplicates("sap_plant").to_dict("records") if F._str(r["sap_plant"])}
    try:
        out.update(_hospitals())
    except HTTPException:                   # spend tables not built
        pass
    return out


def _his_name(plant: str) -> Optional[str]:
    """HIS location name of a SAP plant: from the spend tables, else the hospital map."""
    h = _plant_ids().get(plant)
    return h["his_name"] if h else None


def _as_of(report: Optional[dict], table: Path) -> Optional[str]:
    """When a build ran: its validation report's built_at, else its main table's mtime."""
    if (report or {}).get("built_at"):
        return str(report["built_at"])
    st = F._stamp(table)
    return datetime.fromtimestamp(st[0] / 1e9, tz=timezone.utc).isoformat(timespec="seconds") if st else None


def _built_at() -> Optional[str]:
    return _as_of(_validation(), _data_dir() / f"{SWITCH}.parquet")


def _oof_built_at() -> Optional[str]:
    return _as_of(_read_json(_purchase_dir() / PC.VALIDATION_FILE), _purchase_dir() / f"{PC.OOF}.parquet")


def _scheme_list() -> dict:
    sl = (_validation() or {}).get("scheme_list") or {}
    return {k: sl.get(k) for k in ("file", "financial_year", "switches", "attis_switches", "attis_brands",
                                   "healthium_switches")}


def _cash_list() -> dict:
    cl = (_validation() or {}).get("cash_list") or {}
    b = cl.get("bifurcation") or {}
    return {**{k: cl.get(k) for k in ("file", "financial_year", "switches")},
            "bifurcated_brands": b.get("brands"), "bifurcation_source": b.get("source"),
            "excluded_brands": len((cl.get("excluded") or {}).get("brands") or [])}


def _fy() -> str:
    return _scheme_list().get("financial_year") or "the current"


NOTE_SCOPE = ("Scheme patients are the GOV payer bucket (the same payer rules as Formulary Compliance), IP and "
              "OP. Counted: brands for which HCG's scheme list suggests an Attis brand instead.")
NOTE_SCHEME_SCOPE = ("Scheme patients are the GOV payer bucket (the same payer rules as Formulary Compliance), IP "
                     "and OP. Counted: every brand for which HCG's scheme list suggests a cheaper brand.")
NOTE_HEALTHIUM = ("J&J & Covidien → Healthium is HCG's own line in the report: the scheme list's SUTURES and Endo "
                  "rows (sutures, haemostats, meshes and Covidien staplers).")
NOTE_VIEWS = "The trend ignores the Month filter and the hospital ranking ignores the hospital selector."
NOTE_MONTHS = ("Several months are selected: amounts are summed first and the % is taken once — never an "
               "average of monthly percentages.")
NOTE_CONSUMPTION = ("Attis consumption is every Attis product billed to any payer, at the bill's cost "
                    "(TOTALCOSTPRICE), not list price.")
NOTE_CASH_SCOPE = ("Cash & TPA patients are the CASH and CASH(TPA) payer buckets, IP and OP. Counted: brands for "
                   "which HCG's cash & TPA list suggests a brand that earns HCG more per unit — except the few "
                   "list brands HCG's own report leaves out (app/data/reference/cash_tpa_excluded.csv, to "
                   "confirm with HCG).")
NOTE_CASH_VALUES = ("Cash & TPA: unit margin = MRP ÷ (1 + GST%) − HCG Pharma price, from HCG's FY price list, for "
                    "the brand used and the suggested brand; margin lost = quantity × the difference; % = margin "
                    "lost ÷ the margin the suggested brands would have made (HCG's denominator, unlike the scheme "
                    "% which divides by spend).")
NOTE_BIFURCATION = ("Bifurcated brands: HCG allows them a share (SHARING) of their molecule's volume at the hospital "
                    "in the month, and only the use above that allowance counts. Which brands are bifurcated, and "
                    "their molecule groups, come from HCG's KR Aug-26 report until HCG sends the full rules.")


NOTE_BIFURCATION_COLUMNS = ("Brands table: SHARING, Group qty, Allowed qty and Excess qty are HCG's allowance for the "
                            "hospital and month, worked out on the whole brand (IP + OP, cash + TPA together), so "
                            "they do not follow the Type, Payer or doctor filters; Qty counted does. Unit margin % = "
                            "unit margin ÷ MRP excl GST.")


NOTE_OOF_SCOPE = ("Out of formulary: every goods receipt (GRN, movement 101) of a drug the month's own Item "
                  "Master flags OUT OF FORMULARY, in HCG's purchase scope (app/data/reference/"
                  "purchase_scope_exclusions.csv) — HCG's own out-of-formulary workings, on purchases, not bills.")
NOTE_OOF_VALUES = ("HCG's formulas, per GRN line: margin earned = quantity × (MRP excl GST − Base Price); margin "
                   "possible = the same quantity × the suggested brand's (MRP excl GST − HCG Pharma selling price); "
                   "margin missed = possible − earned, added up NET as HCG does — a line whose brand earns more than "
                   "its suggestion lowers it (the 'current brand better' part); % missed = missed ÷ possible.")
NOTE_OOF_SOURCES = ("Suggested brand: HCG's own worksheet of the month; for a month without one, HCG's pair for the "
                    "brand from its nearest worksheet, then HCG's out-of-formulary list, then the app's match on "
                    "molecule, strength and form in HCG's CASH&TPA list (marked Estimated). Lines HCG parked and "
                    "lines with no suggested brand are shown but add nothing to the margins.")
NOTE_OOF_HCG = ("HCG's figure is the 'Margin Value Difference' of HCG's own Summary sheet for the same hospitals "
                "and months.")
NOTE_OOF_HCG_CUT = ("HCG's Summary is per hospital and month only, so HCG's figure is not shown under a material "
                    "type or brand filter.")
NOTE_WITH_OOF = ("Cash & TPA and out of formulary together, as in HCG's 'CASH(TPA) & OUT OF FORMULARY' table: "
                 "each margin added up, and the % taken on the totals (loss ÷ proposed). The out-of-formulary part "
                 "is HCG's purchase (GRN) method for the same hospitals and months; purchases have no payer or "
                 "patient type, so the Payer and Type filters apply to the cash & TPA part only.")


def _note_not_applied(not_applied: tuple) -> str:
    return (f"Purchases have no payer or patient type: the {' and '.join(not_applied)} "
            f"filter{'s do' if len(not_applied) > 1 else ' does'} not apply here.")


def scope_chips(sc: Scope, *extra: tq.ScopeChip, payer: Optional[str] = None,
                periods: Optional[list] = None) -> tuple:
    """The page filters a list was aggregated over, as table scope chips, then `extra`.
    `periods` are the months the list's data has (default: the spend tables')."""
    chips = [tq.plant_chip(sc.plant), tq.months_chip(sc.months, _periods() if periods is None else periods)]
    if sc.ptype:
        chips.append(tq.chip("Type", "Patient type", sc.ptype))
    if sc.category:
        chips.append(tq.chip("Segment", "Material type", sc.category, CATEGORY_LABELS[sc.category]))
    if sc.material:
        chips.append(tq.chip("Material", "Brand", sc.material))
    if payer:
        chips.append(tq.chip("Payer", "Payer", payer))
    return (*chips, *extra)


def _drill_chips(doctor: Optional[str] = None, group: Optional[str] = None) -> tuple:
    """Chips for the narrowing a list endpoint takes beyond the page filters (the brand is one of
    those: scope_chips says it)."""
    out = []
    if group:
        out.append(tq.chip("Group", "Slice", group, GROUP_LABELS[group]))
    if doctor:
        out.append(tq.chip("Doctor", "Doctor", doctor))
    return tuple(out)


# The lists' own filters (Month, Hospital and Category come from their columns).
_FY_CLASS = tq.Facet("hcg_category", "FY class", order="label")
ITEM_FACETS = (_FY_CLASS,)
CASH_ITEM_FACETS = (_FY_CLASS, tq.Facet("bifurcated", "Bifurcated", value_label=tq.yes_no))
CONSUMPTION_FACETS = (tq.Facet("generic_name", "Generic"),)



def _note_values(attis_only: bool = True) -> str:
    other = "the Attis brand's" if attis_only else "the suggested brand's"
    return (f"Values use HCG's FY {_fy()} price list (HCG Pharma selling price, excl GST), as HCG's report "
            f"does — not the bill's cost: brand used = scheme quantity × its list price; with "
            f"{'Attis' if attis_only else 'the suggested brand'} = the same quantity × {other} list price; "
            "saving missed = the difference; % = saving missed ÷ spend on the brands used.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/spend/meta")
def spend_meta():
    return _meta_cached(_version(), PI._version())


@_cache
def _meta_cached(version: tuple, purchase_version: tuple) -> dict:
    periods = _periods()
    oof_built = _has_oof()
    oof_months = _oof_periods() if oof_built else []
    purchases = PI.months_meta()
    purchase_months = {m["period"] for key in ("prices_months", "generic_months") for m in purchases[key]}
    names = F._sap_names()
    hospitals = [{"plant": p, "name": names.get(p, h["his_name"]), "his_code": h["his_code"],
                  "his_name": h["his_name"]} for p, h in _hospitals().items()]
    hospitals.sort(key=lambda h: h["his_name"])
    return {
        "months": periods,
        "latest": periods[-1] if periods else None,
        "month_labels": {p: F._month_label(p) for p in sorted(set(periods) | set(oof_months) | purchase_months)},
        "hospitals": hospitals,
        "types": list(TYPES),
        "categories": [{"key": k, "label": v} for k, v in CATEGORY_LABELS.items()],
        "payers": list(PAYERS),
        "cash_payers": list(CASH_PAYERS),
        "scheme_list": _scheme_list(),
        "cash_list": _cash_list(),
        # Out of formulary comes from the purchase build: its own months and build time.
        "oof_built": oof_built,
        "oof_months": oof_months,
        "oof_latest": oof_months[-1] if oof_months else None,
        "oof_built_at": _oof_built_at(),
        "oof_suggested_by": [{"key": k, "label": v, "counted": k in COUNTED_SOURCES,
                              "estimated": k == PO.FORMULARY_RULE} for k, v in SUGGESTED_BY.items()],
        # Price variation and generic vs branded: every GRN month, from the same build.
        **purchases,
        "built_at": _built_at(),
        "sources": {
            "sales": "HIS IP + OP pharmacy sales — the formulary build's line cache",
            "scheme_list": "HCG Spend Analysis Main Sheet, sheet SCHEME (suggested brand per brand)",
            "cash_list": "HCG Spend Analysis Main Sheet, sheet CASH&TPA (suggested brand, MRP, GST, SHARING)",
            "bifurcation": "app/data/reference/cash_tpa_bifurcation.csv (seeded from HCG's KR Aug-26 report)",
            "out_of_formulary": "HCG's GRN (goods receipt) extracts, each month's Item Master (OUT OF FORMULARY) "
                                "and HCG's monthly out-of-formulary workbooks: HCG's purchase method "
                                "(python -m app.etl.purchase)",
            "payer_master": "HCG Payer Master + app/data/reference/payer_overrides.csv",
        },
        "notes": [NOTE_SCOPE, _note_values(), NOTE_CONSUMPTION, NOTE_CASH_VALUES, NOTE_BIFURCATION],
    }


@router.get("/spend/attis/overview")
def spend_attis_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                         Segment: str = Query(None), Category: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    return _overview_cached(sc, True, _version())


def _trend_row(p: str, blk: dict) -> dict:
    # `pct` names the % HCG's trend graphs draw beside the amounts (here the saving %).
    return {"period": p, "label": F._month_label(p), **blk, "pct": blk["saving_pct"]}


def _top(frame: pd.DataFrame, cols: list) -> list:
    """HCG's top 10, each with its share of the selection's whole loss."""
    return da._clean_records(frame.head(TOP_N)[cols].rename(columns={"share_pct": "share_of_total_pct"}))


@_cache
def _overview_cached(sc: Scope, attis_only: bool, version: tuple) -> dict:
    sw = _switches(attis_only)
    periods = _periods()
    names = F._sap_names()
    hosp = _hospitals()
    cur = _cut(sw, sc)

    h = _sums(cur, []).iloc[0]
    whole = _sums(_cut(_table(SWITCH), sc), []).iloc[0]
    headline = {**_block(h), "brands": int(cur["material"].nunique()),
                "doctors": int(cur[["plant", "doctor_id"]].drop_duplicates().shape[0]),
                "hospitals": int(cur["plant"].nunique()),
                # this slice's part of the saving missed on HCG's whole scheme list, same scope
                "scheme_saving_missed": F._money(whole["saving_missed"]),
                "share_of_scheme_pct": F._share(h["saving_missed"], whole["saving_missed"])}

    t = _sums(_cut(sw, sc, months=False), ["period"]).set_index("period")
    trend = [_trend_row(p, _block(t.loc[p]) if p in t.index else EMPTY) for p in periods]

    every_unit = _cut(sw, sc, plant=False)
    hb = _sums(every_unit, ["plant"])
    # The ranking's own scope as one figure: what a unit's % is set against (= the All
    # hospitals headline of the same filters).
    network = {"saving_pct": F._pct(_sums(every_unit, []).iloc[0]["saving_pct"])}
    by_hospital = []
    for r in hb.to_dict("records"):
        p = r["plant"]
        hh = hosp.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, hh["his_name"] or p),
                            "his_code": hh["his_code"], "his_name": hh["his_name"], **_block(r),
                            "rank": None, "selected": p == sc.plant})
    # most money missed first
    by_hospital.sort(key=lambda x: (-x["saving_missed"], x["his_name"] or ""))
    for i, x in enumerate(by_hospital):
        x["rank"] = i + 1

    cb = _sums(_cut(sw, sc, category=False), ["category"]).set_index("category")
    by_category = [{"key": k, "label": CATEGORY_LABELS[k],
                    **(_block(cb.loc[c]) if c in cb.index else EMPTY)} for k, c in CATEGORIES.items()]
    tb = _sums(_cut(sw, sc, ptype=False), ["patient_type"]).set_index("patient_type")
    by_type = [{"type": k, **(_block(tb.loc[k]) if k in tb.index else EMPTY)} for k in TYPES]

    top_items = _top(_items_frame(cur), ["material", "material_desc", "suggested_material", "suggested_desc",
                                         "used_value", "suggested_value", "saving_missed", "saving_pct",
                                         "share_pct", "qty"])

    if not attis_only:
        groups = {}
        for g in GROUP_LABELS:
            r = _sums(_cut(_switches(False, g), sc), []).iloc[0]
            groups[g] = {"label": GROUP_LABELS[g], **_block(r),
                         "share_pct": F._share(r["saving_missed"], h["saving_missed"])}
        groups["healthium"].update(_healthium_parts(_cut(_switches(False, "healthium"), sc)))
        notes = [NOTE_SCHEME_SCOPE, _note_values(False), NOTE_VIEWS, NOTE_HEALTHIUM]
        if len(sc.months) > 1:
            notes.append(NOTE_MONTHS)
        if sc.plant and cur.empty:
            notes.append(f"No scheme patient at {names.get(sc.plant, sc.plant)} got a brand with a scheme "
                         f"suggestion in this selection.")
        return {"headline": headline, "scope": _scope_echo(sc, names), "trend": trend,
                "by_hospital": by_hospital, "network": network, "by_category": by_category, "by_type": by_type,
                "top_items": top_items, "groups": groups, "scheme_list": _scheme_list(),
                "notes": notes, "data_as_of": _built_at()}

    use = _cut(_table(ATTIS_USE), sc)
    cost = float(use["cost"].sum())
    consumption = {
        "cost": F._money(cost), "qty": round(float(use["qty"].sum()), 3) + 0.0,
        "lines": int(use["lines"].sum()), "products": int(use["material"].nunique()),
        "by_payer": [{"payer": p, "cost": F._money(c), "share_pct": F._share(c, cost)}
                     for p in PAYERS for c in [float(use.loc[use["payer_bucket"] == p, "cost"].sum())]],
        "by_type": [{"type": t_, "cost": F._money(c), "share_pct": F._share(c, cost)}
                    for t_ in TYPES for c in [float(use.loc[use["patient_type"] == t_, "cost"].sum())]],
        "by_category": [{"key": k, "label": CATEGORY_LABELS[k], "cost": F._money(c),
                         "share_pct": F._share(c, cost)}
                        for k, cat in CATEGORIES.items()
                        for c in [float(use.loc[use["category"] == cat, "cost"].sum())]],
    }

    notes = [NOTE_SCOPE, _note_values(), NOTE_VIEWS, NOTE_CONSUMPTION]
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.plant and cur.empty:
        notes.append(f"No scheme patient at {names.get(sc.plant, sc.plant)} got a brand with an Attis "
                     f"suggestion in this selection.")
    return {
        "headline": headline,
        "scope": _scope_echo(sc, names),
        "trend": trend,
        "by_hospital": by_hospital,
        "network": network,
        "by_category": by_category,
        "by_type": by_type,
        "top_items": top_items,
        "consumption": consumption,
        "scheme_list": _scheme_list(),
        "notes": notes,
        "data_as_of": _built_at(),
    }


def _healthium_parts(d: pd.DataFrame) -> dict:
    """{sutures, endo}: the Healthium line split the way HCG's S6 Summary shows it."""
    if "scheme_group" not in d.columns:             # tables built before Task 3
        d = d.assign(scheme_group=pd.Series(dtype=str))
    g = d["scheme_group"].fillna("").astype(str).str.strip().str.upper()
    out = {}
    for key, (group, label) in HEALTHIUM_PARTS.items():
        r = _sums(d[g == group], []).iloc[0]
        b = _block(r)
        out[key] = {"label": label, **b, "pct": b["saving_pct"]}
    return out


def _scope_echo(sc: Scope, names: dict, payer: Optional[str] = None, not_applied: Optional[tuple] = None) -> dict:
    """The filters a view was cut by. `not_applied` (out of formulary): the filters given but
    meaningless for its data, named instead of a patient type."""
    out = {"plant": sc.plant or "ALL",
           "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
           "his_name": _his_name(sc.plant) if sc.plant else None, "months": list(sc.months)}
    if not_applied is None:
        out["type"] = sc.ptype or "ALL"
    out["category"] = sc.category or "ALL"
    if sc.material:
        out["material"] = sc.material
    if payer is not None:
        out["payer"] = payer
    if not_applied is not None:
        out["not_applied"] = list(not_applied)
    return out


ITEM_COLUMNS = ["material", "material_desc", "suggested_material", "suggested_desc", "category",
                "hcg_category", "used_price", "suggested_price", "qty", "used_value", "suggested_value",
                "saving_missed", "saving_pct", "share_pct", "hospitals", "doctors", "lines"]
# The money and quantity columns each table adds up in its Total row (Contract A `sums`).
ITEM_SUMS = ["qty", "used_value", "suggested_value", "saving_missed", "lines"]


def _items_frame(d: pd.DataFrame) -> pd.DataFrame:
    """One row per brand used: list prices, summed values, the saving and its share."""
    d = d.astype({"material": str, "plant": str})
    r = _sums(d, ["material"])
    info = (F._last(d, ["material"], ["material_desc", "suggested_material", "suggested_desc",
                                       "category", "hcg_category", "used_price", "suggested_price"]))
    spread = d.groupby("material").agg(hospitals=("plant", "nunique"),
                                       doctors=("doctor_id", "nunique")).reset_index()
    r = r.merge(info, on="material", how="left").merge(spread, on="material", how="left")
    tot = float(r["saving_missed"].sum())
    r["share_pct"] = r["saving_missed"] / tot * 100 if tot else np.nan
    r = F._round(r, money=("used_value", "suggested_value", "saving_missed", "used_price",
                           "suggested_price"), pct=("saving_pct", "share_pct"), qty=("qty",))
    r = r.sort_values(["saving_missed", "material"], ascending=[False, True], kind="mergesort")
    return r[ITEM_COLUMNS].reset_index(drop=True)


@router.get("/spend/attis/items")
def spend_attis_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                      Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                      Doctor: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    doctor = F._str(Doctor)
    return F._page(request, _items_cached(sc, True, None, doctor, _version()), ITEM_SUMS, facets=ITEM_FACETS,
                   scope=scope_chips(sc, *_drill_chips(doctor=doctor)))


@_cache
def _items_cached(sc: Scope, attis_only: bool, group: Optional[str], doctor: Optional[str],
                  version: tuple) -> pd.DataFrame:
    d = _cut(_switches(attis_only, group), sc)
    if doctor:
        d = d[d["doctor_id"].astype(str) == doctor]
    return _items_frame(d)


DOCTOR_COLUMNS = ["doctor_id", "doctor_name", "plant", "hospital_name", "brands", "qty", "used_value",
                  "suggested_value", "saving_missed", "saving_pct", "share_pct", "lines"]
DOCTOR_SUMS = ["qty", "used_value", "suggested_value", "saving_missed", "lines"]


@router.get("/spend/attis/doctors")
def spend_attis_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                        Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                        Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    return F._page(request, _doctors_cached(sc, True, None, _version()), DOCTOR_SUMS, scope=scope_chips(sc))


@_cache
def _doctors_cached(sc: Scope, attis_only: bool, group: Optional[str], version: tuple) -> pd.DataFrame:
    d = _cut(_switches(attis_only, group), sc).astype({"plant": str, "doctor_id": str, "material": str})
    keys = ["plant", "doctor_id"]
    r = _sums(d, keys)
    r = r.merge(F._last(d, keys, ["doctor_name", "hospital_name"]), on=keys, how="left")
    r = r.merge(d.groupby(keys).agg(brands=("material", "nunique")).reset_index(), on=keys, how="left")
    tot = float(r["saving_missed"].sum())
    r["share_pct"] = r["saving_missed"] / tot * 100 if tot else np.nan
    r = F._round(r, money=("used_value", "suggested_value", "saving_missed"), pct=("saving_pct", "share_pct"),
                 qty=("qty",))
    r = r.sort_values(["saving_missed", "doctor_id"], ascending=[False, True], kind="mergesort")
    return r[DOCTOR_COLUMNS].reset_index(drop=True)


# ── Scheme: every suggestion (Task 3b) ────────────────────────────────────────
def _group_arg(group) -> Optional[str]:
    return F._choice(group, GROUPS, "Group")


@router.get("/spend/scheme/overview")
def spend_scheme_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                          Segment: str = Query(None), Category: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    return _overview_cached(sc, False, _version())


@router.get("/spend/scheme/items")
def spend_scheme_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                       Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                       Group: str = Query(None), Doctor: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    group, doctor = _group_arg(Group), F._str(Doctor)
    return F._page(request, _items_cached(sc, False, group, doctor, _version()), ITEM_SUMS, facets=ITEM_FACETS,
                   scope=scope_chips(sc, *_drill_chips(doctor=doctor, group=group)))


@router.get("/spend/scheme/doctors")
def spend_scheme_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                         Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                         Group: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    group = _group_arg(Group)
    return F._page(request, _doctors_cached(sc, False, group, _version()), DOCTOR_SUMS,
                   scope=scope_chips(sc, *_drill_chips(group=group)))


# ── Cash & TPA margin loss (Task 3a) ──────────────────────────────────────────
def _cash_payer_arg(payer) -> Optional[str]:
    return F._choice(payer, {p: p for p in CASH_PAYERS}, "Payer")


def _cash_cut(df: pd.DataFrame, sc: Scope, payer: Optional[str], *, payer_filter=True, **kw) -> pd.DataFrame:
    d = _cut(df, sc, **kw)
    if payer and payer_filter:
        d = d[d["payer_bucket"].astype(str) == payer]
    return d


def _cash_sums(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` ([] = one row): summed margins, HCG's loss % (loss / potential), the MRP
    uplift and the qty the bifurcation allowance left out."""
    d = df[CASH_VALUES].astype(float)
    if keys:
        for k in keys:
            d[k] = df[k].astype(str)
        r = d.groupby(keys, sort=True).sum().reset_index()
    else:
        r = d.sum().to_frame().T
    r["loss_pct"] = r["margin_loss"] / r["potential_margin"].where(r["potential_margin"] != 0) * 100
    r["topline_increase"] = r["suggested_mrp_value"] - r["used_mrp_value"]
    r["allowance_qty"] = r["cash_qty"] - r["qty"]
    return r


def _cash_block(r) -> dict:
    return {"actual_margin": F._money(r["actual_margin"]), "potential_margin": F._money(r["potential_margin"]),
            "margin_loss": F._money(r["margin_loss"]), "loss_pct": F._pct(r["loss_pct"]),
            "used_mrp_value": F._money(r["used_mrp_value"]),
            "suggested_mrp_value": F._money(r["suggested_mrp_value"]),
            # HCG's S6 'Used / Suggested brand margin %': margin value ÷ MRP value (excl GST)
            "used_margin_pct": F._share(r["actual_margin"], r["used_mrp_value"]),
            "suggested_margin_pct": F._share(r["potential_margin"], r["suggested_mrp_value"]),
            "topline_increase": F._money(r["topline_increase"]),
            "qty": round(float(r["qty"]), 3) + 0.0, "cash_qty": round(float(r["cash_qty"]), 3) + 0.0,
            "allowance_qty": round(float(r["allowance_qty"]), 3) + 0.0, "lines": int(r["lines"]),
            "bill_cost": F._money(r["actual_cost"])}


CASH_EMPTY = {"actual_margin": 0.0, "potential_margin": 0.0, "margin_loss": 0.0, "loss_pct": None,
              "used_mrp_value": 0.0, "suggested_mrp_value": 0.0, "used_margin_pct": None,
              "suggested_margin_pct": None, "topline_increase": 0.0, "qty": 0.0,
              "cash_qty": 0.0, "allowance_qty": 0.0, "lines": 0, "bill_cost": 0.0}


@router.get("/spend/cash/overview")
def spend_cash_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                        Segment: str = Query(None), Category: str = Query(None), Payer: str = Query(None),
                        Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    return _cash_overview_cached(sc, _cash_payer_arg(Payer), _version())



def _cash_trend_row(p: str, blk: dict) -> dict:
    # actual / potential / loss / pct: the columns of HCG's cash trend table, beside the app's names
    return {"period": p, "label": F._month_label(p), **blk, "actual": blk["actual_margin"],
            "potential": blk["potential_margin"], "loss": blk["margin_loss"], "pct": blk["loss_pct"]}


def _margin_views(t: pd.DataFrame, sc: Scope, payer: Optional[str]) -> dict:
    """The cash overview's views of its margin table: the headline, every month, every
    hospital, and the splits by material type, patient type and payer. `cur` is the selection
    itself."""
    periods = _periods()
    names = F._sap_names()
    hosp = _hospitals()
    cur = _cash_cut(t, sc, payer)

    h = _cash_sums(cur, []).iloc[0]
    headline = {**_cash_block(h), "brands": int(cur["material"].nunique()),
                "doctors": int(cur[["plant", "doctor_id"]].drop_duplicates().shape[0]),
                "hospitals": int(cur["plant"].nunique())}

    tr = _cash_sums(_cash_cut(t, sc, payer, months=False), ["period"]).set_index("period")
    trend = [_cash_trend_row(p, _cash_block(tr.loc[p]) if p in tr.index else CASH_EMPTY) for p in periods]

    every_unit = _cash_cut(t, sc, payer, plant=False)
    hb = _cash_sums(every_unit, ["plant"])
    by_hospital = []
    for r in hb.to_dict("records"):
        p = r["plant"]
        hh = hosp.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, hh["his_name"] or p), "his_code": hh["his_code"],
                            "his_name": hh["his_name"], **_cash_block(r), "rank": None,
                            "selected": p == sc.plant})
    by_hospital.sort(key=lambda x: (-x["margin_loss"], x["his_name"] or ""))
    for i, x in enumerate(by_hospital):
        x["rank"] = i + 1

    cb = _cash_sums(_cash_cut(t, sc, payer, category=False), ["category"]).set_index("category")
    by_category = [{"key": k, "label": CATEGORY_LABELS[k], **(_cash_block(cb.loc[c]) if c in cb.index else CASH_EMPTY)}
                   for k, c in CATEGORIES.items()]
    tb = _cash_sums(_cash_cut(t, sc, payer, ptype=False), ["patient_type"]).set_index("patient_type")
    by_type = [{"type": k, **(_cash_block(tb.loc[k]) if k in tb.index else CASH_EMPTY)} for k in TYPES]
    pb = _cash_sums(_cash_cut(t, sc, payer, payer_filter=False), ["payer_bucket"]).set_index("payer_bucket")
    by_payer = [{"payer": k, **(_cash_block(pb.loc[k]) if k in pb.index else CASH_EMPTY)} for k in CASH_PAYERS]
    # The ranking's own scope as one figure: what a unit's loss % is set against (= the All
    # hospitals headline of the same filters).
    network = {"loss_pct": F._pct(_cash_sums(every_unit, []).iloc[0]["loss_pct"])}
    return {"cur": cur, "headline": headline, "trend": trend, "by_hospital": by_hospital, "network": network,
            "by_category": by_category, "by_type": by_type, "by_payer": by_payer}


@_cache
def _cash_overview_cached(sc: Scope, payer: Optional[str], version: tuple) -> dict:
    v = _margin_views(_table(CASH_SWITCH), sc, payer)
    cur = v["cur"]
    names = F._sap_names()

    top_items = _top(_cash_items_frame(cur), ["material", "material_desc", "suggested_material", "suggested_desc",
                                              "actual_margin", "potential_margin", "margin_loss", "loss_pct",
                                              "share_pct", "qty", "bifurcated"])

    # The allowance is worked out per hospital and month on the whole brand (IP + OP, cash +
    # TPA together, as HCG does), so it follows the hospital, month and material type only.
    al = _cut(_table(CASH_ALLOWANCE).assign(category=lambda x: x["material"].astype(str).str[:1].map(
        {"1": "Onco Drugs", "2": "Other Drugs", "3": "Consumables"})), sc, ptype=False)
    allowance = {"brands": int(al["material"].nunique()),
                 "cash_qty": round(float(al["cash_qty"].sum()), 3) + 0.0,
                 "allowed_qty": round(float(al["allowed_qty"].sum()), 3) + 0.0,
                 "counted_qty": round(float(al["excess_qty"].sum()), 3) + 0.0}

    notes = [NOTE_CASH_SCOPE, NOTE_CASH_VALUES, NOTE_BIFURCATION, NOTE_VIEWS, NOTE_BIFURCATION_COLUMNS]
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.plant and cur.empty:
        notes.append(f"No cash or TPA patient at {names.get(sc.plant, sc.plant)} got a brand with a suggestion "
                     f"in this selection.")
    return {"headline": v["headline"], "scope": _scope_echo(sc, names, payer or "ALL"), "trend": v["trend"],
            "by_hospital": v["by_hospital"], "network": v["network"], "by_category": v["by_category"],
            "by_type": v["by_type"], "by_payer": v["by_payer"], "top_items": top_items, "allowance": allowance,
            "with_oof": _with_oof(sc, payer, cur), "cash_list": _cash_list(), "notes": notes,
            "data_as_of": _built_at()}


def _margin_total(*rows) -> dict:
    """Margin earned, possible and missed added up over summed rows of different margin tables
    (cash & TPA, out of formulary), and the % taken once on the totals."""
    actual, potential, loss = (sum(float(r[c]) for r in rows) for c in ("actual_margin", "potential_margin",
                                                                          "margin_loss"))
    return {"actual_margin": F._money(actual), "potential_margin": F._money(potential),
            "margin_loss": F._money(loss), "loss_pct": F._share(loss, potential)}


def _with_oof(sc: Scope, payer: Optional[str], cash_cur: pd.DataFrame) -> Optional[dict]:
    """HCG's 'CASH(TPA) & OUT OF FORMULARY' table for the selection: the cash row, the
    out-of-formulary purchase row of the same hospitals and months (no payer or patient type
    to cut by), their total, and the same for every month. None until the purchase build has
    run."""
    if not _has_oof():
        return None
    oof = _oof()
    oof_cur = _cut(oof, sc, ptype=False)
    cash, out_of = _cash_sums(cash_cur, []).iloc[0], _oof_sums(oof_cur, []).iloc[0]
    c = _cash_sums(_cash_cut(_table(CASH_SWITCH), sc, payer, months=False), ["period"]).set_index("period")
    o = _oof_sums(_cut(oof, sc, months=False, ptype=False), ["period"]).set_index("period")
    built = set(_oof_periods())
    trend = []
    for p in _periods():
        row = _cash_trend_row(p, _margin_total(*(t.loc[p] for t in (c, o) if p in t.index)))
        row["cash_loss"] = F._money(c.loc[p, "margin_loss"]) if p in c.index else 0.0
        row["oof_loss"] = F._money(o.loc[p, "margin_loss"]) if p in o.index else 0.0
        row["has_oof"] = p in built
        trend.append(row)
    return {"cash": _cash_block(cash), "oof": _oof_block(out_of), "total": _margin_total(cash, out_of),
            "trend": trend, "coverage": _oof_coverage(oof_cur), "oof_months": [m for m in sc.months if m in built],
            "note": NOTE_WITH_OOF}


CASH_ITEM_COLUMNS = ["material", "material_desc", "suggested_material", "suggested_desc", "category",
                     "hcg_category", "used_price", "used_mrp", "used_margin", "used_margin_pct", "suggested_price",
                     "suggested_mrp", "suggested_margin", "suggested_margin_pct", "margin_gap", "cash_qty", "qty",
                     "bifurcated", "sharing", "group_qty", "allowed_qty", "excess_qty", "actual_margin",
                     "potential_margin", "margin_loss", "loss_pct", "share_pct", "topline_increase", "hospitals",
                     "doctors", "lines"]
# HCG's bifurcation columns (S6 CASH&TPA): only a bifurcated brand has them, from cash_allowance.
BIFURCATION_COLUMNS = ["sharing", "group_qty", "allowed_qty", "excess_qty"]
# group_qty is the molecule group's qty, repeated on every brand of that group (MABTAS and
# REDITUX 100MG both carry 83 at KR Aug-26), so a column total would count a group once per
# brand. It gets no Total cell; allowed and excess qty are each brand's own and do add up.
CASH_ITEM_SUMS = ["cash_qty", "qty", "allowed_qty", "excess_qty", "actual_margin", "potential_margin",
                  "margin_loss", "topline_increase", "lines"]


def _cash_items_frame(d: pd.DataFrame) -> pd.DataFrame:
    """One row per brand used: both brands' list price, MRP and unit margin, the qty counted
    after any allowance, the margins and the loss."""
    d = d.astype({"material": str, "plant": str})
    r = _cash_sums(d, ["material"])
    info = F._last(d, ["material"], ["material_desc", "suggested_material", "suggested_desc", "category",
                                     "hcg_category", "used_price", "used_mrp", "used_gst", "suggested_price",
                                     "suggested_mrp", "suggested_gst"])
    used_ex = info["used_mrp"].astype(float) / (1 + info["used_gst"].astype(float) / 100)
    sug_ex = info["suggested_mrp"].astype(float) / (1 + info["suggested_gst"].astype(float) / 100)
    info = info.assign(used_margin=used_ex - info["used_price"].astype(float),
                       suggested_margin=sug_ex - info["suggested_price"].astype(float))
    info["margin_gap"] = info["suggested_margin"] - info["used_margin"]
    # S6's unit margin %: unit margin ÷ unit MRP excl GST
    info["used_margin_pct"] = info["used_margin"] / used_ex.where(used_ex != 0) * 100
    info["suggested_margin_pct"] = info["suggested_margin"] / sug_ex.where(sug_ex != 0) * 100
    spread = d.groupby("material").agg(hospitals=("plant", "nunique"), doctors=("doctor_id", "nunique"),
                                       bifurcated=("bifurcated", "max")).reset_index()
    r = r.merge(info, on="material", how="left").merge(spread, on="material", how="left")
    tot = float(r["margin_loss"].sum())
    r["share_pct"] = r["margin_loss"] / tot * 100 if tot else np.nan
    r["bifurcated"] = r["bifurcated"].fillna(False).astype(bool)
    r = F._round(r, money=("actual_margin", "potential_margin", "margin_loss", "topline_increase", "used_price",
                           "used_mrp", "used_margin", "suggested_price", "suggested_mrp", "suggested_margin",
                           "margin_gap"), pct=("loss_pct", "share_pct", "used_margin_pct", "suggested_margin_pct"),
                 qty=("qty", "cash_qty"))
    r = r.sort_values(["margin_loss", "material"], ascending=[False, True], kind="mergesort")
    return r[[c for c in CASH_ITEM_COLUMNS if c not in BIFURCATION_COLUMNS]].reset_index(drop=True)


def _with_allowance(r: pd.DataFrame, sc: Scope) -> pd.DataFrame:
    """Add HCG's bifurcation columns to the brands table: SHARING and the allowance for the
    selected hospital(s) and month(s). The allowance is worked out on the whole brand (IP + OP,
    cash + TPA together, as HCG does), so it follows the hospital and month only; brands that
    are not bifurcated get null."""
    al = _cut(_table(CASH_ALLOWANCE), sc, ptype=False, category=False).astype({"material": str})
    a = al.groupby("material").agg(group_qty=("group_qty", "sum"), allowed_qty=("allowed_qty", "sum"),
                                   excess_qty=("excess_qty", "sum")).reset_index()
    a = a.merge(F._last(al, ["material"], ["sharing"]), on="material", how="left")
    out = r.merge(a, on="material", how="left")
    out = F._round(out, qty=("group_qty", "allowed_qty", "excess_qty"))
    return out[CASH_ITEM_COLUMNS]


@router.get("/spend/cash/items")
def spend_cash_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                     Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                     Payer: str = Query(None), Doctor: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    payer, doctor = _cash_payer_arg(Payer), F._str(Doctor)
    return F._page(request, _cash_items_cached(sc, payer, doctor, _version()), CASH_ITEM_SUMS,
                   facets=CASH_ITEM_FACETS, scope=scope_chips(sc, *_drill_chips(doctor=doctor), payer=payer))


@_cache
def _cash_items_cached(sc: Scope, payer: Optional[str], doctor: Optional[str], version: tuple) -> pd.DataFrame:
    d = _cash_cut(_table(CASH_SWITCH), sc, payer)
    if doctor:
        d = d[d["doctor_id"].astype(str) == doctor]
    return _with_allowance(_cash_items_frame(d), sc)


CASH_DOCTOR_COLUMNS = ["doctor_id", "doctor_name", "plant", "hospital_name", "brands", "cash_qty", "qty",
                       "actual_margin", "potential_margin", "margin_loss", "loss_pct", "share_pct", "lines"]
CASH_DOCTOR_SUMS = ["cash_qty", "qty", "actual_margin", "potential_margin", "margin_loss", "lines"]


@router.get("/spend/cash/doctors")
def spend_cash_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                       Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                       Payer: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    payer = _cash_payer_arg(Payer)
    return F._page(request, _cash_doctors_cached(sc, payer, _version()), CASH_DOCTOR_SUMS,
                   scope=scope_chips(sc, payer=payer))


@_cache
def _cash_doctors_cached(sc: Scope, payer: Optional[str], version: tuple) -> pd.DataFrame:
    return _margin_doctors(_cash_cut(_table(CASH_SWITCH), sc, payer))


def _margin_doctors(d: pd.DataFrame) -> pd.DataFrame:
    """One row per treating doctor of the cash & TPA margin table's selection."""
    d = d.astype({"plant": str, "doctor_id": str, "material": str})
    keys = ["plant", "doctor_id"]
    r = _cash_sums(d, keys)
    r = r.merge(F._last(d, keys, ["doctor_name", "hospital_name"]), on=keys, how="left")
    r = r.merge(d.groupby(keys).agg(brands=("material", "nunique")).reset_index(), on=keys, how="left")
    tot = float(r["margin_loss"].sum())
    r["share_pct"] = r["margin_loss"] / tot * 100 if tot else np.nan
    r = F._round(r, money=("actual_margin", "potential_margin", "margin_loss"), pct=("loss_pct", "share_pct"),
                 qty=("qty", "cash_qty"))
    r = r.sort_values(["margin_loss", "doctor_id"], ascending=[False, True], kind="mergesort")
    return r[CASH_DOCTOR_COLUMNS].reset_index(drop=True)


# ── Out of formulary: HCG's purchase (GRN) method ─────────────────────────────
# Where a line's suggested brand came from (the purchase build's suggestion_source), or why it
# has none: the coverage tile, the `suggested_by` column and its filter. Only the four sources
# are counted; lines HCG parked (it doubts the Item Master's flag) and lines without a priced
# suggestion are shown and add nothing.
SUGGESTED_BY = {PO.HCG_SHEET: "HCG's worksheet", PO.HCG_SHEET_PAIR: "HCG's pair from another month",
                PO.HCG_LIST: "HCG's out-of-formulary list", PO.FORMULARY_RULE: "Matched by the app (Estimated)",
                PO.PARKED: "Parked by HCG", PO.NONE: "No alternative mapped"}
COUNTED_SOURCES = (PO.HCG_SHEET, PO.HCG_SHEET_PAIR, PO.HCG_LIST, PO.FORMULARY_RULE)
# oof_purchase's columns in the app's words, and what every view adds up. purchase_value is
# HCG's PP of every line (Base Price x qty); counted_value and used_mrp_value are HCG's 'used PP'
# and 'used MRP' (counted lines only); the margins are already 0 on the lines not counted.
OOF_RENAME = {"total_pp": "purchase_value", "total_wo_tax": "value_after_discount",
              "sug_total_pp": "suggested_value", "sug_total_mrp": "suggested_mrp_value",
              "suggested_margin": "potential_margin"}
OOF_MONEY = ["purchase_value", "value_after_discount", "counted_value", "used_mrp_value", "actual_margin",
             "suggested_value", "suggested_mrp_value", "potential_margin", "margin_loss", "gain_loss", "better_loss"]
OOF_VALUES = OOF_MONEY + ["qty", "lines"]


@_cache
def _oof_at(version: tuple) -> pd.DataFrame:
    t = _purchase_table(PC.OOF)
    if t is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT_OOF)
    status = t["status"].astype(str)
    counted = status.eq(PO.COUNTED)
    by = (t["suggestion_source"].astype(str).mask(status.eq(PO.PARKED), PO.PARKED)
          .mask(status.isin((PO.NO_ALTERNATIVE, PO.UNPRICED)), PO.NONE))
    return t.rename(columns=OOF_RENAME).assign(suggested_by=by, counted_value=t["total_pp"].where(counted, 0.0),
                                               used_mrp_value=t["total_mrp"].where(counted, 0.0),
                                               vendor=PP.vendor_keys(t["vendor_name"]))


def _oof() -> pd.DataFrame:
    """oof_purchase in the app's words (OOF_RENAME), with `suggested_by`, HCG's used PP and
    MRP (counted_value, used_mrp_value) and `vendor`: the supplier by name (prices.vendor_keys),
    since one supplier holds several SAP supplier codes (Vardhman Health Specialities has one per
    region) and is one vendor on every view. 503 until the purchase build has run."""
    return _oof_at(_version())


@_cache
def _oof_periods_at(version: tuple) -> list:
    return sorted(set(_oof()["period"].astype(str)))


def _oof_periods() -> list:
    return _oof_periods_at(_version())


def _oof_sums(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` ([] = one row): the summed values and HCG's loss % (missed ÷ possible)."""
    d = df[OOF_VALUES].astype(float)
    if keys:
        for k in keys:
            d[k] = df[k].astype(str)
        r = d.groupby(keys, sort=True).sum().reset_index()
    else:
        r = d.sum().to_frame().T
    r["loss_pct"] = r["margin_loss"] / r["potential_margin"].where(r["potential_margin"] != 0) * 100
    return r


def _oof_block(r) -> dict:
    """One cut's figures. margin_loss is HCG's NET sum (D4); gain_loss + better_loss are its two
    parts, the lines where switching gains and the lines where the brand bought earns more."""
    return {**{c: F._money(r[c]) for c in OOF_MONEY}, "loss_pct": F._pct(r["loss_pct"]),
            # margin ÷ MRP excl GST, for the brands bought and their suggestions (as for cash)
            "used_margin_pct": F._share(r["actual_margin"], r["used_mrp_value"]),
            "suggested_margin_pct": F._share(r["potential_margin"], r["suggested_mrp_value"]),
            "qty": round(float(r["qty"]), 3) + 0.0, "lines": int(r["lines"])}


OOF_EMPTY = _oof_block({**dict.fromkeys(OOF_VALUES, 0.0), "loss_pct": None})


def _oof_coverage(d: pd.DataFrame) -> dict:
    """Where the selection's suggested brands came from, every SUGGESTED_BY key in order: the
    purchase value (HCG's PP), lines, brands and share of the purchase value."""
    total = float(d["purchase_value"].sum())
    g = d.groupby("suggested_by").agg(value=("purchase_value", "sum"), lines=("lines", "sum"),
                                      brands=("material", "nunique"))
    out = {}
    for key, label in SUGGESTED_BY.items():
        value = float(g.at[key, "value"]) if key in g.index else 0.0
        out[key] = {"label": label, "value": F._money(value),
                    "lines": int(g.at[key, "lines"]) if key in g.index else 0,
                    "brands": int(g.at[key, "brands"]) if key in g.index else 0,
                    "share_pct": F._share(value, total), "counted": key in COUNTED_SOURCES,
                    "estimated": key == PO.FORMULARY_RULE}
    return out


class HcgSummary(NamedTuple):
    rows: pd.DataFrame            # HCG's own Summary sheets (oof_hcg_summary): period x centre
    months: frozenset             # the months they cover


def _hcg_summary(sc: Scope) -> Optional[HcgSummary]:
    """HCG's per-centre Summary, to set beside the selection. None when it has not been built,
    or under a Segment or Brand filter: HCG's Summary is per hospital and month only."""
    s = _purchase_table(PC.HCG_SUMMARY)
    if s is None or sc.category or sc.material:
        return None
    return HcgSummary(s, frozenset(s["period"].astype(str)))


def _hcg_check(sc: Scope, hcg: HcgSummary, cur: pd.DataFrame, ours) -> dict:
    """HCG's Summary for the selection beside ours: HCG's margins and loss, our net loss minus
    HCG's, and whether the two agree within the purchase build's guard (PO.MAX_HCG_DIFF for
    each centre and month, a centre on one side only included)."""
    rows = _cut(hcg.rows, sc, ptype=False, category=False)
    pairs = set(zip(rows["period"], rows["plant"])) | set(zip(cur["period"], cur["plant"]))
    used, sug, loss = (float(rows[c].sum()) for c in ("used_margin", "sug_margin", "loss"))
    diff = float(ours["margin_loss"]) - loss
    return {"months": list(sc.months), "centres": int(rows["plant"].nunique()), "actual_margin": F._money(used),
            "potential_margin": F._money(sug), "margin_loss": F._money(loss), "loss_pct": F._share(loss, sug),
            "diff": F._money(diff), "matches": abs(diff) <= PO.MAX_HCG_DIFF * max(len(pairs), 1)}


OOF_TOP_COLUMNS = ["material", "material_desc", "minor_group", "suggested_material", "suggested_desc",
                   "suggested_by", "suggested_by_label", "estimated", "rule", "purchase_value", "actual_margin",
                   "potential_margin", "margin_loss", "loss_pct", "share_pct", "qty"]
OOF_TOP_VENDOR_COLUMNS = ["vendor", "vendor_codes", "vendor_name", "brands", "hospitals", "purchase_value",
                          "actual_margin",
                          "potential_margin", "margin_loss", "loss_pct", "share_pct", "no_alternative_value"]
OOF_NONE_COLUMNS = ["material", "material_desc", "minor_group", "manufacturer", "status", "suggested_material",
                    "suggested_desc", "qty", "purchase_value", "hospitals", "vendors", "lines"]


@router.get("/spend/oof/overview")
def spend_oof_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                       Segment: str = Query(None), Category: str = Query(None), Payer: str = Query(None),
                       Material: str = Query(None)):
    sc, not_applied = _oof_scope(Plant, Month, Type, Segment, Category, Payer, Material)
    return _oof_overview_cached(sc, not_applied, _version())


@_cache
def _oof_overview_cached(sc: Scope, not_applied: tuple, version: tuple) -> dict:
    t = _oof()
    names = F._sap_names()
    ids = _plant_ids()
    cur = _cut(t, sc, ptype=False)
    hcg = _hcg_summary(sc)
    covered = hcg is not None and set(sc.months) <= hcg.months      # HCG has a figure for the whole selection

    h = _oof_sums(cur, []).iloc[0]
    check = _hcg_check(sc, hcg, cur, h) if covered else None
    headline = {**_oof_block(h), "brands": int(cur["material"].nunique()),
                "vendors": int(cur["vendor"].nunique()), "hospitals": int(cur["plant"].nunique()),
                "coverage": _oof_coverage(cur), "hcg_figure": check["margin_loss"] if check else None}

    tr = _oof_sums(_cut(t, sc, months=False, ptype=False), ["period"]).set_index("period")
    hcg_month = (_cut(hcg.rows, sc, months=False, ptype=False, category=False).groupby("period")["loss"].sum()
                 if hcg is not None else None)
    trend = [{**_cash_trend_row(p, _oof_block(tr.loc[p]) if p in tr.index else OOF_EMPTY),
              "hcg_loss": F._money(hcg_month.get(p, 0.0)) if hcg is not None and p in hcg.months else None}
             for p in _oof_periods()]

    every_unit = _cut(t, sc, plant=False, ptype=False)
    hb = _oof_sums(every_unit, ["plant"]).set_index("plant")
    # HCG's figure per centre once its Summary covers the selection: a centre HCG does not list
    # has no loss in HCG's sheet (0), and a centre only HCG lists is still ranked.
    hcg_plant = (_cut(hcg.rows, sc, plant=False, ptype=False, category=False).groupby("plant")["loss"].sum()
                 if covered else None)
    plants = list(hb.index) + ([p for p in hcg_plant.index if p not in hb.index] if covered else [])
    by_hospital = []
    for p in plants:
        pid = ids.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, pid["his_name"] or p), "his_code": pid["his_code"],
                            "his_name": pid["his_name"], **(_oof_block(hb.loc[p]) if p in hb.index else OOF_EMPTY),
                            "hcg_loss": F._money(hcg_plant.get(p, 0.0)) if covered else None,
                            "rank": None, "selected": p == sc.plant})
    # most missed first; a hospital whose brands earned more than the suggestions ranks last
    by_hospital.sort(key=lambda x: (-x["margin_loss"], x["his_name"] or ""))
    for i, x in enumerate(by_hospital):
        x["rank"] = i + 1
    network = {"loss_pct": F._pct(_oof_sums(every_unit, []).iloc[0]["loss_pct"])}

    # The material types purchases have (the purchase scope's drug series), each with its share.
    cb = _oof_sums(_cut(t, sc, category=False, ptype=False), ["category"]).set_index("category")
    present = set(t["category"].astype(str))
    by_category = [{"key": k, "label": CATEGORY_LABELS[k], **(_oof_block(cb.loc[c]) if c in cb.index else OOF_EMPTY)}
                   for k, c in CATEGORIES.items() if c in present]

    items = _oof_items_frame(cur)
    counted = items["suggested_by"].isin(COUNTED_SOURCES)
    better = items[counted & (items["margin_loss"] < 0)].sort_values(["margin_loss", "material"], kind="mergesort")
    none = items[items["suggested_by"].eq(PO.NONE)].sort_values(["purchase_value", "material"],
                                                                ascending=[False, True], kind="mergesort")

    notes = [NOTE_OOF_SCOPE, NOTE_OOF_VALUES, NOTE_OOF_SOURCES, NOTE_VIEWS]
    if hcg is None and (sc.category or sc.material):
        notes.append(NOTE_OOF_HCG_CUT)
    elif hcg is not None:
        notes.append(NOTE_OOF_HCG)
        if not covered:
            notes.append(f"HCG's own Summary covers {', '.join(F._month_label(m) for m in sorted(hcg.months))} only: "
                         f"HCG's figure is shown for those months and not beside a total that includes others.")
    if not_applied:
        notes.append(_note_not_applied(not_applied))
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.plant and cur.empty:
        notes.append(f"No out-of-formulary purchase at {names.get(sc.plant, sc.plant)} in this selection.")
    return {"headline": headline, "hcg": check, "scope": _scope_echo(sc, names, not_applied=not_applied),
            "trend": trend, "by_hospital": by_hospital, "network": network, "by_category": by_category,
            "top_items": _top(items[counted], OOF_TOP_COLUMNS), "better_items": _top(better, OOF_TOP_COLUMNS),
            "top_vendors": _top(_oof_vendors_frame(cur), OOF_TOP_VENDOR_COLUMNS),
            "no_alternative": da._clean_records(none.head(TOP_N)[OOF_NONE_COLUMNS]),
            "notes": notes, "data_as_of": _oof_built_at()}


# One row per OOF brand bought x the brand suggested for it (and where that came from): a brand
# can be partly parked, or have had another suggestion in another month.
OOF_ITEM_KEYS = ["material", "suggested_by", "suggested_material"]
# Descriptive columns, from each row's latest month.
OOF_ITEM_INFO = ["material_desc", "minor_group", "manufacturer", "category", "status", "suggested_desc",
                 "rule_level", "price_source", "sug_price", "sug_mrp", "sug_gst"]
OOF_ITEM_COLUMNS = ["material", "material_desc", "minor_group", "manufacturer", "category", "suggested_by",
                    "suggested_by_label", "estimated", "rule", "status", "suggested_material", "suggested_desc",
                    "price_source", "base_price_min", "base_price_max", "sug_price", "sug_mrp", "sug_gst",
                    "used_margin", "suggested_margin", "margin_gap", "qty", "purchase_value", "value_after_discount",
                    "counted_value", "actual_margin", "suggested_value", "potential_margin", "margin_loss",
                    "loss_pct", "gain_loss", "better_loss", "used_margin_pct", "suggested_margin_pct", "share_pct",
                    "hospitals", "vendors", "lines"]
OOF_ITEM_SUMS = ["qty", "purchase_value", "value_after_discount", "counted_value", "actual_margin",
                 "suggested_value", "potential_margin", "margin_loss", "gain_loss", "better_loss", "lines"]
OOF_ITEM_FACETS = (tq.Facet("suggested_by", "Suggested by", value_label=lambda k: SUGGESTED_BY.get(k, k)),
                   tq.Facet("minor_group", "Molecule"), tq.Facet("manufacturer", "Manufacturer"))


def _rule(level) -> Optional[str]:
    """The app's rule a formulary_rule suggestion was matched by, in words (PO.RULE_LEVELS)."""
    s = F._str(level)
    if s is None:
        return None
    text = PO.RULE_LEVELS.get(s.rstrip(PO.FUZZY_MARK))
    if text and s.endswith(PO.FUZZY_MARK):
        text += " (molecule matched by spelling)"
    return text


def _gain_share(r: pd.DataFrame) -> pd.Series:
    """Each row's part of the margin missed on the lines where switching gains (gain_loss), in
    %. Not of the net figure: a brand earning more than its suggestion can bring the net near
    or below 0, and a share of it would pass 100% or turn negative."""
    tot = float(r["gain_loss"].sum())
    return r["gain_loss"] / tot * 100 if tot else pd.Series(np.nan, index=r.index)


def _oof_items_frame(d: pd.DataFrame) -> pd.DataFrame:
    """One row per OOF brand bought x suggestion: both brands' unit prices, the per-unit margins
    over the counted lines, the summed values and the margins. Counted rows first, most missed
    first; then the parked and unmatched ones, largest purchase first."""
    if d.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in OOF_ITEM_COLUMNS})
    g = d.sort_values("period", kind="mergesort").groupby(OOF_ITEM_KEYS, dropna=False, sort=False)
    r = g.agg(**{c: (c, "last") for c in OOF_ITEM_INFO}, **{c: (c, "sum") for c in OOF_VALUES},
              base_price_min=("base_price_min", "min"), base_price_max=("base_price_max", "max"),
              hospitals=("plant", "nunique"), vendors=("vendor", "nunique")).reset_index()
    counted = r["suggested_by"].isin(COUNTED_SOURCES)
    q = r["qty"].where(counted & (r["qty"] != 0))
    # HCG's unit margins (MRP excl GST - price), qty-weighted over the counted lines
    r["used_margin"] = r["actual_margin"] / q
    r["suggested_margin"] = r["potential_margin"] / q
    r["margin_gap"] = r["margin_loss"] / q
    r["loss_pct"] = r["margin_loss"] / r["potential_margin"].where(r["potential_margin"] != 0) * 100
    r["used_margin_pct"] = r["actual_margin"] / r["used_mrp_value"].where(r["used_mrp_value"] != 0) * 100
    sug_mrp = r["suggested_mrp_value"]
    r["suggested_margin_pct"] = r["potential_margin"] / sug_mrp.where(sug_mrp != 0) * 100
    r["share_pct"] = _gain_share(r)
    r["suggested_by_label"] = r["suggested_by"].map(SUGGESTED_BY)
    r["estimated"] = r["suggested_by"].eq(PO.FORMULARY_RULE)
    r["rule"] = r["rule_level"].map(_rule)
    r = F._round(r, money=("purchase_value", "value_after_discount", "counted_value", "actual_margin",
                           "suggested_value", "potential_margin", "margin_loss", "gain_loss", "better_loss",
                           "used_margin", "suggested_margin", "margin_gap", "base_price_min", "base_price_max",
                           "sug_price", "sug_mrp"),
                 pct=("loss_pct", "used_margin_pct", "suggested_margin_pct", "share_pct", "sug_gst"), qty=("qty",))
    r["_counted"] = counted
    r = r.sort_values(["_counted", "margin_loss", "purchase_value", "material"],
                      ascending=[False, False, False, True], kind="mergesort")
    return r[OOF_ITEM_COLUMNS].reset_index(drop=True)


@router.get("/spend/oof/items")
def spend_oof_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                    Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                    Payer: str = Query(None), Vendor: str = Query(None), Material: str = Query(None)):
    sc, _ = _oof_scope(Plant, Month, Type, Segment, Category, Payer, Material)
    vendor = F._str(Vendor)
    return F._page(request, _oof_items_cached(sc, vendor, _version()), OOF_ITEM_SUMS, facets=OOF_ITEM_FACETS,
                   scope=scope_chips(sc, *_vendor_chips(vendor), periods=_oof_periods()))


def _of_vendor(d: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """The rows of one supplier: Vendor= is its name (any spacing or case, the `vendor` key the
    vendor views send) or one of its SAP supplier codes."""
    return d[d["vendor"].eq(K.norm_text(vendor)) | d["vendor_code"].astype(str).eq(F._code(vendor) or "")]


@_cache
def _oof_items_cached(sc: Scope, vendor: Optional[str], version: tuple) -> pd.DataFrame:
    d = _cut(_oof(), sc, ptype=False)
    if vendor:
        d = _of_vendor(d, vendor)
    return _oof_items_frame(d)


def _vendor_chips(vendor: Optional[str]) -> tuple:
    """Vendor= (one goods supplier's brands) as a chip, named as its latest receipt names it."""
    if not vendor:
        return ()
    hit = _of_vendor(_oof().sort_values("period", kind="mergesort"), vendor)
    return (tq.chip("Vendor", "Vendor", vendor, str(hit["vendor_name"].iloc[-1]) if len(hit) else vendor),)


OOF_VENDOR_COLUMNS = ["vendor", "vendor_codes", "vendor_name", "brands", "hospitals", "qty", "purchase_value",
                      "value_after_discount", "counted_value", "actual_margin", "suggested_value", "potential_margin",
                      "margin_loss", "loss_pct", "gain_loss", "better_loss", "no_alternative_value", "share_pct",
                      "lines"]
OOF_VENDOR_SUMS = ["qty", "purchase_value", "value_after_discount", "counted_value", "actual_margin",
                   "suggested_value", "potential_margin", "margin_loss", "gain_loss", "better_loss",
                   "no_alternative_value", "lines"]


def _oof_vendors_frame(d: pd.DataFrame) -> pd.DataFrame:
    """One row per vendor (the GRN's goods supplier, by name: `vendor`, with every SAP supplier
    code it used in `vendor_codes`) of the selection: its out-of-formulary purchases, the
    margins, and the part of its value with no alternative brand mapped."""
    r = _oof_sums(d, ["vendor"])
    none = d[d["suggested_by"].eq(PO.NONE)].groupby("vendor")["purchase_value"].sum()
    r["no_alternative_value"] = r["vendor"].map(none).fillna(0.0)
    r = r.merge(F._last(d, ["vendor"], ["vendor_name"]), on="vendor", how="left")
    spread = d.groupby("vendor").agg(brands=("material", "nunique"), hospitals=("plant", "nunique"),
                                     vendor_codes=("vendor_code", lambda c: sorted(c.astype(str).unique())))
    r = r.merge(spread.reset_index(), on="vendor", how="left")
    r["share_pct"] = _gain_share(r)
    r["lines"] = r["lines"].astype(int)
    r = F._round(r, money=(*OOF_MONEY, "no_alternative_value"), pct=("loss_pct", "share_pct"), qty=("qty",))
    r = r.sort_values(["margin_loss", "purchase_value", "vendor"], ascending=[False, False, True],
                      kind="mergesort")
    return r[OOF_VENDOR_COLUMNS].reset_index(drop=True)


@router.get("/spend/oof/vendors")
def spend_oof_vendors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                      Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                      Payer: str = Query(None), Material: str = Query(None)):
    sc, _ = _oof_scope(Plant, Month, Type, Segment, Category, Payer, Material)
    return F._page(request, _oof_vendors_cached(sc, _version()), OOF_VENDOR_SUMS,
                   scope=scope_chips(sc, periods=_oof_periods()))


@_cache
def _oof_vendors_cached(sc: Scope, version: tuple) -> pd.DataFrame:
    return _oof_vendors_frame(_cut(_oof(), sc, ptype=False))


CONSUMPTION_COLUMNS = ["material", "material_desc", "generic_name", "category", "qty", "cost", "gov_cost",
                       "ip_cost", "op_cost", "share_pct", "hospitals", "lines"]
CONSUMPTION_SUMS = ["qty", "cost", "gov_cost", "ip_cost", "op_cost", "lines"]


@router.get("/spend/attis/consumption")
def spend_attis_consumption(request: Request, Plant: str = Query(None), Month: str = Query(None),
                            Type: str = Query(None), Segment: str = Query(None),
                            Category: str = Query(None), Payer: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category)
    payer = F._choice(Payer, {p: p for p in PAYERS}, "Payer")
    return F._page(request, _consumption_cached(sc, payer, _version()), CONSUMPTION_SUMS, facets=CONSUMPTION_FACETS,
                   scope=scope_chips(sc, payer=payer))


@_cache
def _consumption_cached(sc: Scope, payer: Optional[str], version: tuple) -> pd.DataFrame:
    d = _cut(_table(ATTIS_USE), sc).astype({"material": str, "plant": str})
    if payer:
        d = d[d["payer_bucket"].astype(str) == payer]
    cost = d["cost"].astype(float)
    d = d.assign(gov_cost=cost.where(d["payer_bucket"] == "GOV", 0.0),
                 ip_cost=cost.where(d["patient_type"] == "IP", 0.0),
                 op_cost=cost.where(d["patient_type"] == "OP", 0.0))
    r = d.groupby("material").agg(qty=("qty", "sum"), cost=("cost", "sum"), gov_cost=("gov_cost", "sum"),
                                  ip_cost=("ip_cost", "sum"), op_cost=("op_cost", "sum"),
                                  hospitals=("plant", "nunique"), lines=("lines", "sum")).reset_index()
    r = r.merge(F._last(d, ["material"], ["material_desc", "generic_name", "category"]), on="material",
                how="left")
    tot = float(r["cost"].sum())
    r["share_pct"] = r["cost"] / tot * 100 if tot else np.nan
    r = F._round(r, money=("cost", "gov_cost", "ip_cost", "op_cost"), pct=("share_pct",), qty=("qty",))
    r = r.sort_values(["cost", "material"], ascending=[False, True], kind="mergesort")
    return r[CONSUMPTION_COLUMNS].reset_index(drop=True)


# ── Brand filter ──────────────────────────────────────────────────────────────
# The page's Brand filter picks one brand used (Material=, which every endpoint above takes);
# this lists the brands it offers. Each comes with what HCG's lists would replace it by in each
# section and what that switch would have added: the margin for cash & TPA patients and for
# out-of-formulary purchases, the spend saved for scheme patients. The sections stay apart, as
# everywhere on the page: a scheme spend and a cash margin are different measures.
BRAND_SECTIONS = ("scheme", "cash", "oof")
BRAND_LIMIT = 20
BRAND_MAX = 50


def _brand_side(d: pd.DataFrame, sums: pd.DataFrame, loss: str, pct: str) -> pd.DataFrame:
    """One section's row per brand used: its suggested brand (the one it was most often set
    against) and the rupees and % the switch would have added."""
    sug = (d.astype({"material": str}).groupby(["material", "suggested_material", "suggested_desc"], dropna=False)
           ["lines"].sum().reset_index().sort_values(["material", "lines"], ascending=[True, False], kind="mergesort")
           .drop_duplicates("material")[["material", "suggested_material", "suggested_desc"]])
    r = sums[["material", loss, pct, "qty"]].rename(columns={loss: "missed", pct: "pct"})
    return r.merge(sug, on="material", how="left")


@_cache
def _brands_cached(sc: Scope, payer: Optional[str], version: tuple) -> list:
    scheme = _cut(_switches(False), sc)
    cash = _cash_cut(_table(CASH_SWITCH), sc, payer)
    sides = {
        "scheme": _brand_side(scheme, _sums(scheme, ["material"]), "saving_missed", "saving_pct"),
        "cash": _brand_side(cash, _cash_sums(cash, ["material"]), "margin_loss", "loss_pct"),
    }
    used = [scheme, cash]
    if _has_oof():
        # Out of formulary is purchases: every OOF brand bought for the same hospitals, months and
        # material type (no payer or patient type), its suggestion from the counted lines. A brand
        # with no alternative is offered too, missing nothing, as the section shows it.
        oof = _cut(_oof(), sc, ptype=False)
        sides["oof"] = _brand_side(oof[oof["suggested_by"].isin(COUNTED_SOURCES)], _oof_sums(oof, ["material"]),
                                   "margin_loss", "loss_pct")
        used.append(oof)
    names = F._last(pd.concat([t[["period", "material", "material_desc"]].astype({"material": str})
                               for t in used], ignore_index=True), ["material"], ["material_desc"])
    out = {str(m): {"material": str(m), "material_desc": desc, **{k: None for k in BRAND_SECTIONS}}
           for m, desc in zip(names["material"], names["material_desc"])}
    for key, side in sides.items():
        for r in side.itertuples(index=False):
            out[str(r.material)][key] = {
                "suggested_material": F._code(r.suggested_material), "suggested_desc": F._str(r.suggested_desc),
                "missed": F._money(r.missed), "pct": F._pct(r.pct), "qty": round(float(r.qty), 3) + 0.0}
    # The biggest switch first, by the largest rupees any one section would have gained.
    best = lambda b: max((b[k]["missed"] for k in BRAND_SECTIONS if b[k]), default=0.0)
    return sorted(out.values(), key=lambda b: (-best(b), b["material_desc"] or "", b["material"]))


def _brand_match(q: Optional[str]):
    """`q` matches a code's start or every word of the name, case apart."""
    if not q:
        return lambda b: True
    words = q.lower().split()
    return lambda b: (b["material"].startswith(q.strip())
                      or all(w in (b["material_desc"] or "").lower() for w in words))


@router.get("/spend/brands")
def spend_brands(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                 Segment: str = Query(None), Category: str = Query(None), Payer: str = Query(None),
                 Section: str = Query(None), Material: str = Query(None), q: str = Query(None),
                 limit: int = Query(BRAND_LIMIT)):
    """The brands the page's Brand filter offers under its other filters, biggest switch first;
    `q` searches them, `Material` returns that one brand. `Section` (scheme / cash / oof) keeps
    only the brands used in that section, ranked by its own gain: what a section's tab can show."""
    sc = _scope(Plant, Month, Type, Segment, Category)
    payer, material, text = _cash_payer_arg(Payer), F._code(Material), F._str(q)
    section = F._choice(Section, {k.upper(): k for k in BRAND_SECTIONS}, "Section")
    rows = _brands_cached(sc, payer, _version())
    if section:
        rows = sorted((b for b in rows if b[section]), key=lambda b: (-b[section]["missed"], b["material_desc"] or "", b["material"]))
    hits = [b for b in rows if b["material"] == material] if material else [b for b in rows if _brand_match(text)(b)]
    n = max(1, min(int(limit) if isinstance(limit, int) else BRAND_LIMIT, BRAND_MAX))
    return {"brands": hits[:n], "total": len(hits), "scope": _scope_echo(sc, F._sap_names(), payer or "ALL")}


@router.get("/spend/validation", dependencies=[Depends(require_admin)])
def spend_validation():
    """The build's own report: the scheme list read, per-month totals, HCG's KR check."""
    v = _validation()
    if v is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return v
