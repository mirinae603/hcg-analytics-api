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
                                       top brands, the bifurcation allowance
    GET /spend/cash/items           -> {data,total} per brand used; with Doctor=, one doctor's
    GET /spend/cash/doctors         -> {data,total} per doctor; with Material=, one brand's
    GET /spend/oof/overview         -> out-of-formulary margin loss (cash & TPA patients): the
                                       same views as cash, plus how much of the OOF use found a
                                       formulary brand to compare with
    GET /spend/oof/items            -> {data,total} per OOF brand used, with its matched brand
    GET /spend/oof/doctors          -> {data,total} per doctor; with Material=, one brand's
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

Out of formulary (HCG's 'CASH(TPA) & OUT OF FORMULARY' table): cash and TPA use of a brand the
month's Item Master flags OUT OF FORMULARY, set against the formulary brand of the same
molecule, strength and form with the best margin in HCG's CASH&TPA price list (the build's own
match, app/etl/spend/oof.py, until HCG sends its list). Actual margin = bill MRP excl GST - bill
cost; proposed = qty x the matched brand's unit margin; % = loss / proposed, as for cash. The
cash overview adds the two together the way HCG's table does (`with_oof`).

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
from app.core import data_access as da
from app.core import table_query as tq
from app.core.config import settings
from app.core.deps import require_admin

router = APIRouter()

SWITCH, ATTIS_USE = "scheme_switch", "attis_consumption"
CASH_SWITCH, CASH_ALLOWANCE = "cash_switch", "cash_allowance"
OOF_SWITCH = "oof_switch"
TABLES = (SWITCH, ATTIS_USE, CASH_SWITCH, CASH_ALLOWANCE, OOF_SWITCH)
VALIDATION_FILE = "spend_validation.json"
NOT_BUILT = "Spend data has not been built yet — run python -m app.etl.spend"
NOT_BUILT_OOF = ("The out-of-formulary margin has not been built yet — run python -m app.etl.spend again to "
                 "add it")

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
# Read at CALL time from settings.SPEND_DATA_DIR and keyed on each file's mtime/size, like
# the formulary tables: a rebuild or a test's own fixtures are picked up without a restart.
def _data_dir() -> Path:
    return Path(settings.SPEND_DATA_DIR)


@lru_cache(maxsize=8)
def _read(path: str, stamp: tuple) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("attis", "bifurcated", "matched"):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df


def _table(name: str) -> pd.DataFrame:
    p = _data_dir() / f"{name}.parquet"
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT_OOF if name == OOF_SWITCH else NOT_BUILT)
    df = _read(str(p), stamp)
    if name == OOF_SWITCH and "margin_loss" not in df.columns:      # written before its margins were
        raise HTTPException(status_code=503, detail=NOT_BUILT_OOF)
    return df


def _has(name: str) -> bool:
    try:
        _table(name)
    except HTTPException:
        return False
    return True


def _version() -> tuple:
    d = _data_dir()
    return (str(d),) + tuple(F._stamp(d / f"{t}.parquet") for t in TABLES) + (F._stamp(d / VALIDATION_FILE),)


def _validation() -> Optional[dict]:
    try:
        return json.loads((_data_dir() / VALIDATION_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None


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


def _months_arg(month, periods: list) -> tuple:
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
            F._bad(f"No spend data for {m}. Available months: {', '.join(periods)}")
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


def _his_name(plant: str) -> Optional[str]:
    """HIS location name of a SAP plant: from the spend tables, else the hospital map."""
    h = _hospitals().get(plant)
    if h:
        return h["his_name"]
    hm = da.hospital_map()
    m = hm[hm["sap_plant"].astype(str) == plant]
    return str(m["his_location_name"].iloc[0]) if len(m) else None


def _built_at() -> Optional[str]:
    v = _validation() or {}
    if v.get("built_at"):
        return str(v["built_at"])
    st = F._stamp(_data_dir() / f"{SWITCH}.parquet")
    return datetime.fromtimestamp(st[0] / 1e9, tz=timezone.utc).isoformat(timespec="seconds") if st else None


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


NOTE_OOF_SCOPE = ("Out of formulary: cash and TPA patients' use of brands the month's Item Master flags OUT OF "
                  "FORMULARY (the same items as the Formulary Compliance page), IP and OP.")
NOTE_OOF_MATCH = ("HCG has not sent which formulary brand should replace each out-of-formulary brand, so the app "
                  "finds one in HCG's own masters: a formulary brand with the same molecule (Item Master 'Minor "
                  "Group'), the same strength and the same form, priced in HCG's CASH&TPA list; of several, the "
                  "one with the best margin. Every match is listed in the Brands table so HCG can check it; HCG's "
                  "own list replaces it once sent.")
NOTE_OOF_VALUES = ("Actual margin = the bill's MRP excl GST − the bill's cost (an out-of-formulary brand has no FY "
                   "list price; the GST is the matched brand's); proposed margin = the same quantity × the matched "
                   "brand's unit margin (MRP excl GST − HCG Pharma price); loss = proposed − actual; % = loss ÷ "
                   "proposed, as for cash. Brands with no match are listed but add nothing.")
NOTE_WITH_OOF = ("Cash & TPA and out of formulary together, as in HCG's 'CASH(TPA) & OUT OF FORMULARY' table: "
                 "each margin added up, and the % taken on the totals (loss ÷ proposed).")


def scope_chips(sc: Scope, *extra: tq.ScopeChip, payer: Optional[str] = None) -> tuple:
    """The page filters a list was aggregated over, as table scope chips, then `extra`."""
    chips = [tq.plant_chip(sc.plant), tq.months_chip(sc.months, _periods())]
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
OOF_ITEM_FACETS = (tq.Facet("matched", "Matched", value_label=tq.yes_no),
                   tq.Facet("suggested_category", "Formulary brand class", order="label"),
                   tq.Facet("molecule", "Molecule"))
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
    return _meta_cached(_version())


@_cache
def _meta_cached(version: tuple) -> dict:
    periods = _periods()
    names = F._sap_names()
    hospitals = [{"plant": p, "name": names.get(p, h["his_name"]), "his_code": h["his_code"],
                  "his_name": h["his_name"]} for p, h in _hospitals().items()]
    hospitals.sort(key=lambda h: h["his_name"])
    return {
        "months": periods,
        "latest": periods[-1] if periods else None,
        "month_labels": {p: F._month_label(p) for p in periods},
        "hospitals": hospitals,
        "types": list(TYPES),
        "categories": [{"key": k, "label": v} for k, v in CATEGORY_LABELS.items()],
        "payers": list(PAYERS),
        "cash_payers": list(CASH_PAYERS),
        "scheme_list": _scheme_list(),
        "cash_list": _cash_list(),
        "oof_built": _has(OOF_SWITCH),
        "built_at": _built_at(),
        "sources": {
            "sales": "HIS IP + OP pharmacy sales — the formulary build's line cache",
            "scheme_list": "HCG Spend Analysis Main Sheet, sheet SCHEME (suggested brand per brand)",
            "cash_list": "HCG Spend Analysis Main Sheet, sheet CASH&TPA (suggested brand, MRP, GST, SHARING)",
            "bifurcation": "app/data/reference/cash_tpa_bifurcation.csv (seeded from HCG's KR Aug-26 report)",
            "out_of_formulary": "each month's Item Master (OUT OF FORMULARY, Minor Group) + the CASH&TPA sheet's "
                                "prices: the build's own molecule, strength and form match",
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


def _scope_echo(sc: Scope, names: dict, payer: Optional[str] = None) -> dict:
    out = {"plant": sc.plant or "ALL",
           "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
           "his_name": _his_name(sc.plant) if sc.plant else None, "months": list(sc.months),
           "type": sc.ptype or "ALL", "category": sc.category or "ALL"}
    if sc.material:
        out["material"] = sc.material
    if payer is not None:
        out["payer"] = payer
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
    """What the cash and out-of-formulary overviews share, from one margin table: the
    headline, every month, every hospital, and the splits by material type, patient type and
    payer. `cur` is the selection itself."""
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


def _matched(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["matched"].astype(bool)]


def _oof_coverage(d: pd.DataFrame) -> dict:
    """How much of the out-of-formulary use in the selection found a formulary brand. A brand
    matched in some selected months counts as matched; its unmatched months' cost is not."""
    m = _matched(d)
    cost, mcost = float(d["actual_cost"].sum()), float(m["actual_cost"].sum())
    brands, matched = int(d["material"].nunique()), int(m["material"].nunique())
    return {"brands": brands, "matched_brands": matched, "unmatched_brands": brands - matched,
            "bill_cost": F._money(cost), "matched_cost": F._money(mcost),
            "unmatched_cost": F._money(cost - mcost), "matched_cost_pct": F._share(mcost, cost) if cost else None}


def _with_oof(sc: Scope, payer: Optional[str], cash_cur: pd.DataFrame) -> Optional[dict]:
    """HCG's 'CASH(TPA) & OUT OF FORMULARY' table for the selection: the cash row, the
    out-of-formulary row and their total, plus the total for every month. None until the
    out-of-formulary table has been built."""
    if not _has(OOF_SWITCH):
        return None
    oof = _table(OOF_SWITCH)
    om = _matched(oof)
    oof_cur = _cash_cut(om, sc, payer)
    keep = CASH_VALUES + ["period"]
    total = _cash_sums(pd.concat([cash_cur[keep], oof_cur[keep]], ignore_index=True), []).iloc[0]
    cs_all = _cash_cut(_table(CASH_SWITCH), sc, payer, months=False)
    oof_all = _cash_cut(om, sc, payer, months=False)
    t = _cash_sums(pd.concat([cs_all[keep], oof_all[keep]], ignore_index=True), ["period"]).set_index("period")
    c = _cash_sums(cs_all, ["period"]).set_index("period")
    o = _cash_sums(oof_all, ["period"]).set_index("period")
    trend = []
    for p in _periods():
        row = _cash_trend_row(p, _cash_block(t.loc[p]) if p in t.index else CASH_EMPTY)
        row["cash_loss"] = F._money(c.loc[p, "margin_loss"]) if p in c.index else 0.0
        row["oof_loss"] = F._money(o.loc[p, "margin_loss"]) if p in o.index else 0.0
        trend.append(row)
    return {"cash": _cash_block(_cash_sums(cash_cur, []).iloc[0]), "oof": _cash_block(_cash_sums(oof_cur, []).iloc[0]),
            "total": _cash_block(total), "trend": trend, "coverage": _oof_coverage(_cash_cut(oof, sc, payer)),
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
    """One row per treating doctor of a margin table's selection (cash or out of formulary)."""
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


# ── Out of formulary (cash & TPA patients) ────────────────────────────────────
OOF_ITEM_COLUMNS = ["material", "material_desc", "molecule", "suggested_material", "suggested_desc", "matched",
                    "margin_loss", "loss_pct", "actual_margin", "potential_margin", "category", "suggested_category",
                    "candidates", "qty", "actual_cost", "bill_mrp", "used_gst", "used_margin", "used_margin_pct",
                    "suggested_price", "suggested_mrp", "suggested_margin", "suggested_margin_pct", "margin_gap",
                    "share_pct", "hospitals", "doctors", "lines"]
OOF_ITEM_SUMS = ["qty", "actual_cost", "bill_mrp", "actual_margin", "potential_margin", "margin_loss", "lines"]


def _oof_items_frame(d: pd.DataFrame) -> pd.DataFrame:
    """One row per out-of-formulary brand used: its bill, the formulary brand matched to it
    with that brand's list price and margin, and the margins. Brands without a match keep
    their bill values and no margins, after the matched ones."""
    d = d.astype({"material": str, "plant": str})
    if d.empty:
        return pd.DataFrame({c: pd.Series(dtype=bool if c == "matched" else object) for c in OOF_ITEM_COLUMNS})
    m = _matched(d)
    r = _cash_sums(m, ["material"]) if len(m) else pd.DataFrame(columns=["material", *CASH_VALUES, "loss_pct"])
    r = r[["material", "actual_margin", "potential_margin", "margin_loss", "loss_pct", "used_mrp_value", "qty"]]
    r = r.rename(columns={"qty": "matched_qty"})
    b = d.groupby("material").agg(qty=("qty", "sum"), actual_cost=("actual_cost", "sum"), bill_mrp=("bill_mrp", "sum"),
                                  lines=("lines", "sum"), hospitals=("plant", "nunique"),
                                  doctors=("doctor_id", "nunique"), matched=("matched", "max")).reset_index()
    # The match itself from the brand's latest MATCHED month (a brand can be matched in one month
    # and not the next); brands never matched keep their latest row.
    pick = pd.concat([m, d[~d["material"].isin(m["material"])]])
    info = F._last(d, ["material"], ["material_desc", "molecule", "category"]).merge(
        F._last(pick, ["material"], ["suggested_material", "suggested_desc", "suggested_category", "candidates",
                                     "used_gst", "suggested_price", "suggested_mrp", "suggested_gst"]),
        on="material", how="left")
    out = b.merge(r, on="material", how="left").merge(info, on="material", how="left")
    mq = out["matched_qty"].astype(float)
    mq = mq.where(mq != 0)
    sug_ex = out["suggested_mrp"].astype(float) / (1 + out["suggested_gst"].astype(float) / 100)
    out["used_margin"] = out["actual_margin"].astype(float) / mq
    out["suggested_margin"] = sug_ex - out["suggested_price"].astype(float)
    out["margin_gap"] = out["suggested_margin"] - out["used_margin"]
    um = out["used_mrp_value"].astype(float)
    out["used_margin_pct"] = out["actual_margin"].astype(float) / um.where(um != 0) * 100
    out["suggested_margin_pct"] = out["suggested_margin"] / sug_ex.where(sug_ex != 0) * 100
    tot = float(out["margin_loss"].astype(float).sum())
    out["share_pct"] = out["margin_loss"].astype(float) / tot * 100 if tot else np.nan
    out["matched"] = out["matched"].astype(bool)
    out = F._round(out, money=("actual_margin", "potential_margin", "margin_loss", "actual_cost", "bill_mrp",
                               "used_margin", "suggested_price", "suggested_mrp", "suggested_margin", "margin_gap"),
                   pct=("loss_pct", "share_pct", "used_margin_pct", "suggested_margin_pct", "used_gst"), qty=("qty",))
    out["_loss"] = out["margin_loss"].astype(float).fillna(-np.inf)
    out = out.sort_values(["matched", "_loss", "actual_cost", "material"], ascending=[False, False, False, True],
                          kind="mergesort")
    return out[OOF_ITEM_COLUMNS].reset_index(drop=True)


@router.get("/spend/oof/overview")
def spend_oof_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                       Segment: str = Query(None), Category: str = Query(None), Payer: str = Query(None),
                       Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    return _oof_overview_cached(sc, _cash_payer_arg(Payer), _version())


@_cache
def _oof_overview_cached(sc: Scope, payer: Optional[str], version: tuple) -> dict:
    t = _table(OOF_SWITCH)
    v = _margin_views(_matched(t), sc, payer)
    names = F._sap_names()
    every = _cash_cut(t, sc, payer)
    items = _oof_items_frame(every)
    hit = items["matched"].astype(bool)
    top_items = _top(items[hit], ["material", "material_desc", "molecule", "suggested_material",
                                               "suggested_desc", "suggested_category", "actual_margin",
                                               "potential_margin", "margin_loss", "loss_pct", "share_pct", "qty"])
    unmatched = da._clean_records(items[~hit].sort_values("actual_cost", ascending=False)
                                  .head(TOP_N)[["material", "material_desc", "molecule", "qty", "actual_cost"]])
    notes = [NOTE_OOF_SCOPE, NOTE_OOF_MATCH, NOTE_OOF_VALUES, NOTE_VIEWS]
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.plant and every.empty:
        notes.append(f"No cash or TPA patient at {names.get(sc.plant, sc.plant)} got an out-of-formulary brand "
                     f"in this selection.")
    return {"headline": {**v["headline"], "coverage": _oof_coverage(every)},
            "scope": _scope_echo(sc, names, payer or "ALL"), "trend": v["trend"], "by_hospital": v["by_hospital"],
            "network": v["network"],
            "by_category": v["by_category"], "by_type": v["by_type"], "by_payer": v["by_payer"],
            "top_items": top_items, "unmatched": unmatched, "notes": notes, "data_as_of": _built_at()}


@router.get("/spend/oof/items")
def spend_oof_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                    Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                    Payer: str = Query(None), Doctor: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    payer, doctor = _cash_payer_arg(Payer), F._str(Doctor)
    return F._page(request, _oof_items_cached(sc, payer, doctor, _version()), OOF_ITEM_SUMS,
                   facets=OOF_ITEM_FACETS, scope=scope_chips(sc, *_drill_chips(doctor=doctor), payer=payer))


@_cache
def _oof_items_cached(sc: Scope, payer: Optional[str], doctor: Optional[str], version: tuple) -> pd.DataFrame:
    d = _cash_cut(_table(OOF_SWITCH), sc, payer)
    if doctor:
        d = d[d["doctor_id"].astype(str) == doctor]
    return _oof_items_frame(d)


@router.get("/spend/oof/doctors")
def spend_oof_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                      Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                      Payer: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Material)
    payer = _cash_payer_arg(Payer)
    return F._page(request, _oof_doctors_cached(sc, payer, _version()), CASH_DOCTOR_SUMS,
                   scope=scope_chips(sc, payer=payer))


@_cache
def _oof_doctors_cached(sc: Scope, payer: Optional[str], version: tuple) -> pd.DataFrame:
    return _margin_doctors(_cash_cut(_matched(_table(OOF_SWITCH)), sc, payer))


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
# section and what that switch would have added: the margin for cash & TPA and out-of-formulary
# patients, the spend saved for scheme patients. The sections stay apart, as everywhere on the
# page: a scheme spend and a cash margin are different measures.
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
    oof = _cash_cut(_matched(_table(OOF_SWITCH)), sc, payer) if _has(OOF_SWITCH) else _table(CASH_SWITCH).iloc[0:0]
    sides = {
        "scheme": _brand_side(scheme, _sums(scheme, ["material"]), "saving_missed", "saving_pct"),
        "cash": _brand_side(cash, _cash_sums(cash, ["material"]), "margin_loss", "loss_pct"),
        "oof": _brand_side(oof, _cash_sums(oof, ["material"]), "margin_loss", "loss_pct"),
    }
    names = F._last(pd.concat([t[["period", "material", "material_desc"]].astype({"material": str})
                               for t in (scheme, cash, oof)], ignore_index=True), ["material"], ["material_desc"])
    out = {str(m): {"material": str(m), "material_desc": desc, **{k: None for k in BRAND_SECTIONS}}
           for m, desc in zip(names["material"], names["material_desc"])}
    for key, side in sides.items():
        for r in side.itertuples(index=False):
            out[str(r.material)][key] = {
                "suggested_material": F._code(r.suggested_material), "suggested_desc": r.suggested_desc,
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
