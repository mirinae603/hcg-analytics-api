"""Formulary Compliance — HCG's monthly formulary compliance report.

Served from the small aggregated tables `python -m app.etl.formulary` publishes into
settings.FORMULARY_DATA_DIR (line-level sales never reach the repo or this process):

    GET /formulary/meta           -> months, hospitals, filter options, sources
    GET /formulary/overview       -> headline %, trend, segments, IP/OP, hospital ranking and
                                     the all-hospitals figure beside it (network),
                                     class mix, GOV scheme views, HCG's onco / non-onco
                                     consumption table, consumption by payer
    GET /formulary/doctors        -> {data,total} per treating doctor; with Material=, the
                                     doctors behind one item
    GET /formulary/items          -> {data,total} items of one formulary group; with
                                     Doctor=, one doctor's items
    GET /formulary/scheme/items   -> {data,total} GOV-billed restricted / preferred brands
    GET /formulary/validation     -> the build's validation report (admin only)

Compliance = Formulary (A) / (Formulary (A) + Out of Formulary (B)) on drug cost. Non
Formulary (C) is outside the ratio and consumables are outside it altogether. Every
percentage is a ratio of SUMS taken after filtering — never an average of percentages
across hospitals or months — and is None when its denominator is 0.

The list endpoints also return `sums` (data_access.paginate) over the filtered rows, so every
share column has its amount on screen, and take the shared table filters (app.core.table_query:
f.<field>= facets such as formulary class or manufacturer, literal search; facets=1 lists them
with counts and echoes the page filters as `scope`).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core import data_access as da
from app.core import table_query as tq
from app.core.config import settings
from app.core.deps import require_admin

router = APIRouter()

TABLES = ("formulary_summary", "formulary_by_doctor", "formulary_by_item", "formulary_doctor_item",
          "formulary_rule_breach", "formulary_rules")
VALIDATION_FILE = "formulary_validation.json"
# HCG's PAN MIS, read by `python -m app.etl.formulary.hcg_mis`: each unit's budget. Optional —
# until it is built every hospital's budget is TARGET_PCT.
MIS_COMPLIANCE = "hcg_mis_compliance"
MIS_VALIDATION_FILE = "hcg_mis_validation.json"
NOT_BUILT = "Formulary data has not been built yet — run python -m app.etl.formulary"

# The app's existing category names (data_access.CATEGORIES), so resolve_category keeps
# working: the build derives them from the material code's first digit (1 / 2 / 3).
ONCO, NON_ONCO = "Onco Drugs", "Other Drugs"
DRUGS = (ONCO, NON_ONCO)
SEGMENTS = {"onco": ONCO, "non-onco": NON_ONCO}
SEGMENT_LABELS = {"onco": "Onco", "non-onco": "Non-Onco"}
_SEGMENT_IN = {"ONCO": "onco", "NON-ONCO": "non-onco", "NON ONCO": "non-onco",
               "NON_ONCO": "non-onco", "NONONCO": "non-onco"}
_CATEGORY_OF_DIGIT = {"1": ONCO, "2": NON_ONCO, "3": "Consumables"}
TYPES = ("IP", "OP")
PAYERS = ("CASH", "CASH(TPA)", "GOV")
# HCG switched to pooled with the Jun-26 MIS (sent 13 Jul 2026), restating Apr/May-26 the same
# way; the May-26 mail's first figures and the whole FY 25-26 series are the mean method.
METHODS = {"pooled": "Pooled (HCG's current method)",
           "mean": "Mean of Onco & Non-Onco (HCG's earlier method)"}
# HCG's MIS "Budget" default. The PAN MIS sets each unit's own (85% to 100% in FY 26-27).
TARGET_PCT = 100
# Scheme views: what HCG's report calls "restricted for scheme" / "preferred for scheme",
# counted on GOV-billed drug lines only.
SCHEME_CLASSES = {"restricted": ("FORMULARYR", "FORMULARYR-OPD"), "preferred": ("FORMULARYP",)}
SCHEME_PAYER = "GOV"
GROUPS = {"A": "formulary", "B": "out_of_formulary", "C": "non_formulary"}
GROUP_LABELS = {"A": "Formulary", "B": "Out of formulary", "C": "Non formulary"}
SCHEME_KIND_LABELS = {"restricted": "Restricted for scheme", "preferred": "Preferred for scheme"}
CONSUMABLES = "Consumables"

_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# ── Tables ────────────────────────────────────────────────────────────────────
# Read at CALL time from settings.FORMULARY_DATA_DIR (never a path captured at import), and
# keyed on each file's mtime/size: a rebuild — or a test pointing the setting at its own
# fixtures — is picked up without a restart, and a stale frame is never served.
def _data_dir() -> Path:
    return Path(settings.FORMULARY_DATA_DIR)


def _stamp(p: Path):
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


@lru_cache(maxsize=8)
def _read(path: str, stamp: tuple) -> pd.DataFrame:
    return pd.read_parquet(path)


def _table(name: str) -> pd.DataFrame:
    p = _data_dir() / f"{name}.parquet"
    stamp = _stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return _read(str(p), stamp)


def _version() -> tuple:
    """Identity of the published data — part of every result-cache key."""
    d = _data_dir()
    return ((str(d),) + tuple(_stamp(d / f"{t}.parquet") for t in TABLES) + (_stamp(d / VALIDATION_FILE),)
            + (_stamp(d / f"{MIS_COMPLIANCE}.parquet"), _stamp(d / MIS_VALIDATION_FILE)))


def _mis_source() -> Optional[dict]:
    """The PAN MIS workbook the budgets were read from: file, its month, when it was read."""
    try:
        r = json.loads((_data_dir() / MIS_VALIDATION_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None
    return {"file": r.get("source_file"), "period": r.get("file_period"), "built_at": r.get("built_at")}


def _validation() -> Optional[dict]:
    p = _data_dir() / VALIDATION_FILE
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return None


# Results are pure functions of the published tables, so they are memoized per normalised
# argument tuple (+ _version()). POST /admin/refresh-data clears them via clear_caches().
_RESULT_CACHES: list = []


def _cache(fn):
    cached = lru_cache(maxsize=64)(fn)
    _RESULT_CACHES.append(cached)
    return cached


def clear_caches() -> None:
    _read.cache_clear()
    for c in _RESULT_CACHES:
        c.cache_clear()


def plants_with_data() -> set:
    """SAP plants present in formulary_summary (empty when the data is not built) — the
    'formulary' domain in /meta/plants."""
    try:
        return set(_table("formulary_summary")["plant"].astype(str).unique())
    except HTTPException:
        return set()


# ── Arguments ─────────────────────────────────────────────────────────────────
class Scope(NamedTuple):
    plant: Optional[str]          # SAP plant code; None => all hospitals
    months: tuple                 # periods 'YYYY-MM'
    ptype: Optional[str]          # 'IP' / 'OP'; None => both
    segment: Optional[str]        # 'onco' / 'non-onco'; None => both
    payer: Optional[str]          # 'CASH' / 'CASH(TPA)' / 'GOV'; None => all
    method: str                   # 'pooled' / 'mean'


def _bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def _str(v) -> Optional[str]:
    # Handlers are also called directly as Python functions (app/ai/kpi_registry), where an
    # untouched Query(None) default is a fastapi Query object, not None.
    if not isinstance(v, str):
        return None
    v = v.strip()
    return v or None


def _choice(value, allowed: dict, name: str) -> Optional[str]:
    """Canonical value for a filter, None for ALL / empty, 400 for anything else."""
    s = _str(value)
    if s is None or s.upper() == "ALL":
        return None
    out = allowed.get(s.upper())
    if out is None:
        _bad(f"Invalid {name} '{s}'. Use one of: ALL, {', '.join(dict.fromkeys(allowed.values()))}")
    return out


def _plant_arg(plant) -> Optional[str]:
    """SAP plant code for Plant= (None => all hospitals). A value that does not resolve is
    a 404 here, never the silent all-hospitals answer da.resolve_plant gives elsewhere."""
    s = _str(plant)
    if s is None or s.upper() in ("ALL", "ALL PLANTS", "ALL HOSPITALS"):
        return None
    code = da.resolve_plant(s)
    if not code:
        raise HTTPException(status_code=404, detail=(
            f"Unknown hospital '{s}'. Use a SAP plant code (e.g. HC05), a HIS code "
            f"(e.g. KABHK) or a hospital name."))
    return code


def _periods() -> list:
    return sorted(_table("formulary_summary")["period"].astype(str).unique())


def _months_arg(month, periods: list) -> tuple:
    s = _str(month)
    if s is None:
        return tuple(periods[-1:])          # latest period, like HCG's monthly report
    if s.upper() == "ALL":
        return tuple(periods)
    out = []
    for m in (x.strip() for x in s.split(",")):
        if not m:
            continue
        if not _PERIOD_RE.match(m):
            _bad(f"Invalid Month '{m}'. Use YYYY-MM, a comma list of them, or ALL")
        if m not in periods:
            _bad(f"No formulary data for {m}. Available months: {', '.join(periods)}")
        out.append(m)
    if not out:
        _bad("Month is empty. Use YYYY-MM, a comma list of them, or ALL")
    return tuple(sorted(set(out)))


def _segment_arg(segment, category) -> Optional[str]:
    """Segment=onco|non-onco, or the app-wide Category= (Onco Drugs / Other Drugs)."""
    seg = _choice(segment, _SEGMENT_IN, "Segment")
    cat = da.resolve_category(_str(category))
    if cat:
        by_cat = {v: k for k, v in SEGMENTS.items()}.get(cat)
        if by_cat is None:
            _bad(f"Category '{category}' has no formulary compliance — it covers drugs only. "
                 f"Use Onco Drugs or Other Drugs (or Segment=onco / non-onco)")
        if seg and seg != by_cat:
            _bad(f"Segment '{segment}' and Category '{category}' disagree")
        seg = by_cat
    return seg


def _method_arg(method) -> str:
    s = _str(method)
    if s is None:
        return "pooled"
    if s.lower() not in METHODS:
        _bad(f"Invalid Method '{s}'. Use pooled or mean")
    return s.lower()


def _scope(Plant, Month, Type, Segment, Category, Payer, Method=None) -> Scope:
    periods = _periods()                    # 503 first: nothing below means anything without data
    return Scope(
        plant=_plant_arg(Plant),
        months=_months_arg(Month, periods),
        ptype=_choice(Type, {t: t for t in TYPES}, "Type"),
        segment=_segment_arg(Segment, Category),
        payer=_choice(Payer, {p: p for p in PAYERS}, "Payer"),
        method=_method_arg(Method),
    )


def _code(v) -> Optional[str]:
    """A material code as the tables store it ('101313', not 101313.0)."""
    s = _str(v)
    return re.sub(r"\.0+$", "", s) if s else None


# ── Filtering + arithmetic ────────────────────────────────────────────────────
def _category(df: pd.DataFrame) -> pd.Series:
    if "category" in df.columns:
        return df["category"].astype(str)
    # formulary_doctor_item carries no category: same first-digit rule the build uses.
    return df["material"].astype(str).str[:1].map(_CATEGORY_OF_DIGIT).fillna("")


def _cut(df: pd.DataFrame, sc: Scope, *, plant=True, months=True, ptype=True, payer=True,
         drugs=True) -> pd.DataFrame:
    """Rows of `df` in scope. The flags drop one filter for the views that ignore it."""
    m = pd.Series(True, index=df.index)
    if plant and sc.plant:
        m &= df["plant"].astype(str) == sc.plant
    if months:
        m &= df["period"].astype(str).isin(sc.months)
    if ptype and sc.ptype:
        m &= df["patient_type"].astype(str) == sc.ptype
    cat = _category(df)
    if sc.segment:
        m &= cat == SEGMENTS[sc.segment]
    elif drugs:
        m &= cat.isin(DRUGS)
    if payer and sc.payer:
        m &= df["payer_bucket"].astype(str) == sc.payer
    return df[m]


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    den = a + b
    return a / den.where(den != 0) * 100


def _rollup(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` ([] = one row): A / B / C cost, all drug cost (and its onco / non-onco
    split) and lines, and the ratios —
    pooled over both segments, per segment, and the mean of the segment ratios (a segment
    whose A + B is 0 is skipped)."""
    cost = df["cost"].astype(float)
    grp = df["formulary_group"].astype(str)
    cat = _category(df)
    d = pd.DataFrame({
        "formulary": cost.where(grp == "A", 0.0),
        "out_of_formulary": cost.where(grp == "B", 0.0),
        "non_formulary": cost.where(grp == "C", 0.0),
        "total": cost,
        "onco": cost.where(cat == ONCO, 0.0),
        "non_onco": cost.where(cat == NON_ONCO, 0.0),
        "lines": df["lines"].astype("int64"),
        "_oa": cost.where((grp == "A") & (cat == ONCO), 0.0),
        "_ob": cost.where((grp == "B") & (cat == ONCO), 0.0),
        "_na": cost.where((grp == "A") & (cat == NON_ONCO), 0.0),
        "_nb": cost.where((grp == "B") & (cat == NON_ONCO), 0.0),
    }, index=df.index)
    if keys:
        for k in keys:
            d[k] = df[k].astype(str)
        r = d.groupby(keys, sort=True).sum().reset_index()
    else:
        r = d.sum().to_frame().T
    r["pooled_pct"] = _ratio(r["formulary"], r["out_of_formulary"])
    r["onco_pct"] = _ratio(r["_oa"], r["_ob"])
    r["non_onco_pct"] = _ratio(r["_na"], r["_nb"])
    r["mean_pct"] = r[["onco_pct", "non_onco_pct"]].mean(axis=1)
    return r.drop(columns=["_oa", "_ob", "_na", "_nb"])


def _money(v) -> float:
    return round(float(v), 2) + 0.0         # + 0.0 turns -0.0 into 0.0


def _pct(v) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    return None if f != f else round(f, 4)


def _share(part, whole) -> Optional[float]:
    return _pct(float(part) / float(whole) * 100) if float(whole) else None


def _split(r, method: str) -> dict:
    """The A / B / C block every breakdown shares; compliance_pct follows Method."""
    return {"compliance_pct": _pct(r["mean_pct"] if method == "mean" else r["pooled_pct"]),
            "formulary": _money(r["formulary"]),
            "out_of_formulary": _money(r["out_of_formulary"]),
            "non_formulary": _money(r["non_formulary"]),
            "lines": int(r["lines"])}


# HCG's month heading ('Aug-26'), the same one the table filters label their months with.
_month_label = tq.month_label


def _fy(period: str) -> str:
    """'2026-08' -> '2026-27' (HCG's financial year runs April to March)."""
    y, m = int(period[:4]), int(period[5:7])
    s = y if m >= 4 else y - 1
    return f"{s}-{(s + 1) % 100:02d}"


@_cache
def _budget_book(version: tuple) -> dict:
    """Each unit's budget from HCG's PAN MIS: by (plant, month), and by (plant, FY) for the
    months the MIS does not list (the FY's latest). Empty when the MIS has not been built."""
    p = _data_dir() / f"{MIS_COMPLIANCE}.parquet"
    stamp = _stamp(p)
    if stamp is None:
        return {}
    by_month, by_fy = {}, {}
    for r in _read(str(p), stamp).dropna(subset=["budget_pct"]).sort_values("period").itertuples():
        by_month[(r.plant, r.period)] = float(r.budget_pct)
        by_fy[(r.plant, _fy(r.period))] = float(r.budget_pct)
    return {"month": by_month, "fy": by_fy}


def _budget(plant: Optional[str], periods) -> tuple:
    """(budget %, source) of one hospital over `periods`. HCG sets budgets per unit, so all
    hospitals together have none; nor does a unit outside HCG's PAN list, a FY the MIS does not
    cover, or a span whose months' budgets differ. Before the MIS is built: TARGET_PCT."""
    if plant is None or not periods:
        return None, None
    book = _budget_book(_version())
    if not book:
        return float(TARGET_PCT), "default"
    vals = set()
    for p in periods:
        b = book["month"].get((plant, p))
        vals.add(book["fy"].get((plant, _fy(p))) if b is None else b)
    if len(vals) != 1 or None in vals:
        return None, None
    return vals.pop(), "hcg_mis"


@_cache
def _hospitals_at(version: tuple) -> dict:
    """SAP plant -> {his_code, his_name} as the latest period's sales name it."""
    s = _table("formulary_summary").sort_values("period")
    last = s.drop_duplicates("plant", keep="last")
    return {str(r["plant"]): {"his_code": str(r["hospital"]), "his_name": str(r["hospital_name"])}
            for r in last[["plant", "hospital", "hospital_name"]].to_dict("records")}


def _hospitals() -> dict:
    return _hospitals_at(_version())


def _his_name(plant: str) -> Optional[str]:
    """HIS location name of a SAP plant: from the sales tables, else the hospital map."""
    h = _hospitals().get(plant)
    if h:
        return h["his_name"]
    hm = da.hospital_map()
    m = hm[hm["sap_plant"].astype(str) == plant]
    return str(m["his_location_name"].iloc[0]) if len(m) else None


def _sap_names() -> dict:
    try:
        dp = da.load("dim_plant")
    except FileNotFoundError:
        return {}
    return {str(c): str(n) for c, n in zip(dp["plant"], dp["plant_name"])}


def _built_at() -> Optional[str]:
    v = _validation() or {}
    if v.get("built_at"):
        return str(v["built_at"])
    st = _stamp(_data_dir() / "formulary_summary.parquet")
    return datetime.fromtimestamp(st[0] / 1e9, tz=timezone.utc).isoformat(timespec="seconds") if st else None


def _month_reports(v: Optional[dict]) -> dict:
    """period -> that month's section of the validation report."""
    if not isinstance(v, dict):
        return {}
    m = v.get("months", v)
    if isinstance(m, list):
        return {str(x["period"]): x for x in m if isinstance(x, dict) and x.get("period")}
    if isinstance(m, dict):
        return {str(k): x for k, x in m.items() if _PERIOD_RE.match(str(k)) and isinstance(x, dict)}
    return {}


NOTE_RATIO = ("Compliance = Formulary ÷ (Formulary + Out of Formulary), on drug cost. Non Formulary "
              "is outside the ratio; consumables are excluded.")
NOTE_METHOD = {
    "pooled": "Pooled: one ratio over onco and non-onco drug spend together (HCG's current method).",
    "mean": ("Mean: the average of the onco ratio and the non-onco ratio (HCG's earlier method); a segment "
             "with no formulary or out-of-formulary spend is left out."),
}
NOTE_SEGMENT = "Onco / Non-Onco follow the material code's first digit (1 = onco, 2 = non-onco)."
NOTE_VIEWS = ("The trend ignores the Month filter, the hospital ranking ignores the hospital selector, and the "
              "scheme views count GOV-billed drug lines only, whatever the Payer filter.")
NOTE_MONTHS = ("Several months are selected: amounts are summed first and the ratio is taken once — never an "
               "average of monthly percentages.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/formulary/meta")
def formulary_meta():
    return _meta_cached(_version())


@_cache
def _meta_cached(version: tuple) -> dict:
    periods = _periods()
    names = _sap_names()
    hospitals = [{"plant": p, "name": names.get(p, h["his_name"]), "his_code": h["his_code"],
                  "his_name": h["his_name"]} for p, h in _hospitals().items()]
    hospitals.sort(key=lambda h: h["his_name"])
    reports = _month_reports(_validation())
    return {
        "months": periods,
        "latest": periods[-1] if periods else None,
        "month_labels": {p: _month_label(p) for p in periods},
        "hospitals": hospitals,
        "types": list(TYPES),
        "segments": [{"key": k, "label": v} for k, v in SEGMENT_LABELS.items()],
        "payers": list(PAYERS),
        "methods": [{"key": k, "label": v} for k, v in METHODS.items()],
        "target_pct": TARGET_PCT,
        # Which workbook the unit budgets come from (None until the MIS is built).
        "hcg_mis": _mis_source(),
        "built_at": _built_at(),
        "sources": {
            "sales": "HIS IP + OP pharmacy sales — HCG's monthly consolidated files",
            "item_master": "HCG Item Master of the same month",
            "payer_master": "HCG Payer Master + app/data/reference/payer_overrides.csv",
            "hospital_map": "app/data/reference/hospital_map.csv",
            # file, sheet, rows and totals per month — the scalar fields of the build's report
            "months": {p: {k: x for k, x in rep.items()
                           if isinstance(x, (str, int, float)) and not isinstance(x, bool)}
                       for p, rep in sorted(reports.items())},
        },
        "notes": [NOTE_RATIO, NOTE_METHOD["pooled"], NOTE_METHOD["mean"], NOTE_SEGMENT],
    }


@router.get("/formulary/overview")
def formulary_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                       Segment: str = Query(None), Category: str = Query(None),
                       Payer: str = Query(None), Method: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Payer, Method)
    return _overview_cached(sc, _version())


@_cache
def _overview_cached(sc: Scope, version: tuple) -> dict:
    s = _table("formulary_summary")
    periods = _periods()
    names = _sap_names()
    hosp = _hospitals()
    method = sc.method
    drugs = _cut(s, sc)

    h = _rollup(drugs, []).iloc[0]
    sp = _split(h, method)
    headline = {"compliance_pct": sp["compliance_pct"], "pooled_pct": _pct(h["pooled_pct"]),
                "mean_pct": _pct(h["mean_pct"]), "onco_pct": _pct(h["onco_pct"]),
                "non_onco_pct": _pct(h["non_onco_pct"]), "formulary": sp["formulary"],
                "out_of_formulary": sp["out_of_formulary"], "non_formulary": sp["non_formulary"],
                "total": _money(h["total"]), "lines": sp["lines"],
                # the ratio's denominator (A + B) and HCG's MIS budget, beside the %
                "in_ratio": _money(h["formulary"] + h["out_of_formulary"])}
    headline["budget_pct"], headline["budget_source"] = _budget(sc.plant, sc.months)

    t = _rollup(_cut(s, sc, months=False), ["period"]).set_index("period")
    trend = []
    for p in periods:
        r = t.loc[p] if p in t.index else None
        row = {"period": p, "label": _month_label(p)}
        if r is None:
            row.update({"compliance_pct": None, "pooled_pct": None, "mean_pct": None, "formulary": 0.0,
                        "out_of_formulary": 0.0, "non_formulary": 0.0, "lines": 0})
        else:
            sp = _split(r, method)
            row.update({"compliance_pct": sp["compliance_pct"], "pooled_pct": _pct(r["pooled_pct"]),
                        "mean_pct": _pct(r["mean_pct"]), "formulary": sp["formulary"],
                        "out_of_formulary": sp["out_of_formulary"],
                        "non_formulary": sp["non_formulary"], "lines": sp["lines"]})
        # HCG's MIS table per month: Budget beside Actual, and the amounts behind the %.
        row.update({"budget_pct": _budget(sc.plant, [p])[0],
                    "in_ratio": _money(row["formulary"] + row["out_of_formulary"]),
                    "total": _money(row["formulary"] + row["out_of_formulary"] + row["non_formulary"])})
        trend.append(row)

    seg = _rollup(drugs.assign(_seg=_category(drugs)), ["_seg"]).set_index("_seg")
    segments = []
    for key, cat in SEGMENTS.items():
        r = seg.loc[cat] if cat in seg.index else None
        # Within one segment pooled and mean are the same ratio.
        sp = _split(r, "pooled") if r is not None else {
            "compliance_pct": None, "formulary": 0.0, "out_of_formulary": 0.0, "non_formulary": 0.0, "lines": 0}
        segments.append({"key": key, "label": SEGMENT_LABELS[key], **sp})

    ty = _rollup(drugs, ["patient_type"]).set_index("patient_type")
    by_type = [{"type": k, **(_split(ty.loc[k], method) if k in ty.index else {
        "compliance_pct": None, "formulary": 0.0, "out_of_formulary": 0.0, "non_formulary": 0.0, "lines": 0})}
        for k in TYPES]

    every_unit = _cut(s, sc, plant=False)
    hb = _rollup(every_unit, ["plant"])
    by_hospital = []
    for r in hb.to_dict("records"):
        p = r["plant"]
        hh = hosp.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, hh["his_name"] or p),
                            "his_code": hh["his_code"], "his_name": hh["his_name"],
                            **_split(r, method), "budget_pct": _budget(p, sc.months)[0],
                            "rank": None, "selected": p == sc.plant})
    # Worst first; a hospital with no A + B spend has no ratio, so no rank, and goes last.
    by_hospital.sort(key=lambda x: (x["compliance_pct"] is None,
                                    x["compliance_pct"] if x["compliance_pct"] is not None else 0.0,
                                    -x["out_of_formulary"], x["his_name"] or ""))
    for i, x in enumerate(b for b in by_hospital if b["compliance_pct"] is not None):
        x["rank"] = i + 1

    total = float(drugs["cost"].sum())
    cl = (drugs.groupby(["formulary_class", "formulary_group"], as_index=False)[["cost", "lines"]].sum()
          .sort_values(["formulary_group", "cost"], ascending=[True, False]))
    by_class = [{"formulary_class": r["formulary_class"], "formulary_group": r["formulary_group"],
                 "cost": _money(r["cost"]), "lines": int(r["lines"]), "share_pct": _share(r["cost"], total)}
                for r in cl.to_dict("records")]

    # Scheme views: GOV lines only, whatever the Payer filter; shares are of the drug
    # spend of ALL payers in the same scope (per segment for the segment rows).
    every_payer = _cut(s, sc, payer=False)
    cat_all = _category(every_payer)
    tot_all = float(every_payer["cost"].sum())
    tot_seg = {k: float(every_payer.loc[cat_all == c, "cost"].sum()) for k, c in SEGMENTS.items()}
    gov = every_payer[every_payer["payer_bucket"].astype(str) == SCHEME_PAYER]

    def scheme_figure(k: pd.DataFrame) -> dict:
        kc = _category(k)
        block = {"cost": _money(k["cost"].sum()), "lines": int(k["lines"].sum()),
                 "share_pct": _share(k["cost"].sum(), tot_all)}
        for key, c in SEGMENTS.items():
            v = float(k.loc[kc == c, "cost"].sum())
            block[key.replace("-", "_")] = {"cost": _money(v), "share_pct": _share(v, tot_seg[key])}
        return block

    scheme: dict = {kind: scheme_figure(gov[gov["formulary_class"].astype(str).isin(classes)])
                    for kind, classes in SCHEME_CLASSES.items()}
    # HCG's third brand figure: restricted brands billed to cash & TPA patients (Task 4).
    non_gov = every_payer[every_payer["payer_bucket"].astype(str) != SCHEME_PAYER]
    scheme["restricted_cash_tpa"] = scheme_figure(
        non_gov[non_gov["formulary_class"].astype(str).isin(SCHEME_CLASSES["restricted"])])
    scheme["gov_total"] = _money(gov["cost"].sum())
    # The amounts every share above divides by (all payers' drug spend in the same scope).
    scheme["base"] = {"all": _money(tot_all), "onco": _money(tot_seg["onco"]),
                      "non_onco": _money(tot_seg["non-onco"])}

    notes = [NOTE_RATIO, NOTE_METHOD[method], NOTE_SEGMENT, NOTE_VIEWS]
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.segment and method == "mean":
        notes.append("Only one segment is selected, so the mean method equals that segment's own ratio.")
    if sc.plant and drugs.empty:
        notes.append(f"No formulary drug spend for {names.get(sc.plant, sc.plant)} in this selection.")

    return {
        "headline": headline,
        "scope": {"plant": sc.plant or "ALL",
                  "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
                  "his_name": _his_name(sc.plant) if sc.plant else None, "months": list(sc.months),
                  "type": sc.ptype or "ALL", "segment": sc.segment or "ALL",
                  "payer": sc.payer or "ALL", "method": method},
        "trend": trend,
        "segments": segments,
        "by_type": by_type,
        "by_hospital": by_hospital,
        # The same figure over every hospital (the ranking's own scope): the reference a unit is
        # set against. Equals the All-hospitals headline of the same filters.
        "network": {"compliance_pct": _split(_rollup(every_unit, []).iloc[0], method)["compliance_pct"]},
        "by_class": by_class,
        "scheme": scheme,
        "consumption": _consumption(s, sc),
        "by_payer": _by_payer(s, sc),
        "notes": notes,
        "data_as_of": _built_at(),
    }


NOTE_CONSUMPTION = (
    "HCG's 'Onco and Non Onco Formulary Consumption' table, on drug cost (consumables excluded). Every % here "
    "is the value ÷ that row's total consumption (Formulary + Out of formulary + Non formulary), so non "
    "formulary IS in this denominator — unlike the compliance %, which is Formulary ÷ (Formulary + Out of "
    "formulary). Scheme = GOV-billed lines; cash & TPA = CASH and CASH(TPA). The table shows every payer and "
    "both segments whatever the Payer and Segment filters; Month, hospital and IP/OP apply.")


def _consumption(s: pd.DataFrame, sc: Scope) -> dict:
    """HCG's spend-email table: per segment (and both together) F / B / C, their total, and
    restricted / preferred brand spend for scheme and restricted for cash & TPA, each with
    its share of that segment's total consumption."""
    d = _cut(s, sc._replace(segment=None), payer=False)          # drugs, every payer
    cat = _category(d)
    cls = d["formulary_class"].astype(str)
    grp = d["formulary_group"].astype(str)
    gov = d["payer_bucket"].astype(str) == SCHEME_PAYER
    restricted = cls.isin(SCHEME_CLASSES["restricted"])
    parts = {"formulary": grp == "A", "out_of_formulary": grp == "B", "non_formulary": grp == "C",
             "restricted_scheme": restricted & gov,
             "preferred_scheme": cls.isin(SCHEME_CLASSES["preferred"]) & gov,
             "restricted_cash_tpa": restricted & ~gov}
    pct_name = {"formulary": "formulary_pct", "out_of_formulary": "oof_pct", "non_formulary": "nf_pct"}
    rows = []
    for key, label, seg in (("onco", "Onco", cat == ONCO), ("non-onco", "Non-Onco", cat == NON_ONCO),
                            ("all", "Total", cat.isin(DRUGS))):
        v = {k: float(d.loc[seg & m, "cost"].sum()) for k, m in parts.items()}
        total = v["formulary"] + v["out_of_formulary"] + v["non_formulary"]
        row = {"segment": key, "label": label}
        for k in ("formulary", "out_of_formulary", "non_formulary"):
            row[k] = _money(v[k])
        row["total"] = _money(total)
        for k in ("formulary", "out_of_formulary", "non_formulary"):
            row[pct_name[k]] = _share(v[k], total)
        for k in ("restricted_scheme", "preferred_scheme", "restricted_cash_tpa"):
            row[k] = _money(v[k])
            row[f"{k}_pct"] = _share(v[k], total)
        rows.append(row)
    return {"scope_note": NOTE_CONSUMPTION, "rows": rows}


def _by_payer(s: pd.DataFrame, sc: Scope) -> dict:
    """The spend email's payer-wise consumption (consumables included) and the layout of HCG's
    S5 'Summary' pivot (drug cost by class x payer, onco / non-onco x IP / OP). Every payer,
    both segments, whatever the Payer and Segment filters."""
    d = _cut(s, sc._replace(segment=None), payer=False, drugs=False)
    cat = _category(d)
    pay = d["payer_bucket"].astype(str)
    ptype = d["patient_type"].astype(str)
    cost = d["cost"].astype(float)

    def figures(m: pd.Series) -> dict:
        c = cost[m]
        onco, non_onco = float(c[cat == ONCO].sum()), float(c[cat == NON_ONCO].sum())
        consumables = float(c[cat == CONSUMABLES].sum())
        drugs = d[m & cat.isin(DRUGS)]
        r = _rollup(drugs, []).iloc[0]
        return {"onco": _money(onco), "non_onco": _money(non_onco), "consumables": _money(consumables),
                "drugs": _money(onco + non_onco), "total": _money(onco + non_onco + consumables),
                "onco_ip": _money(c[(cat == ONCO) & (ptype == "IP")].sum()),
                "onco_op": _money(c[(cat == ONCO) & (ptype == "OP")].sum()),
                "non_onco_ip": _money(c[(cat == NON_ONCO) & (ptype == "IP")].sum()),
                "non_onco_op": _money(c[(cat == NON_ONCO) & (ptype == "OP")].sum()),
                **{k: v for k, v in _split(r, sc.method).items() if k != "lines"}}

    rows = [{"payer": p, **figures(pay == p)} for p in PAYERS]
    rows.append({"payer": "ALL", **figures(pd.Series(True, index=d.index))})
    whole = rows[-1]["total"]
    for r in rows:
        r["share_pct"] = _share(r["total"], whole)

    drugs = d[cat.isin(DRUGS)]
    dc = _category(drugs).map({ONCO: "onco", NON_ONCO: "non_onco"})
    key = dc + "_" + drugs["patient_type"].astype(str).str.lower()
    cells = ["onco_ip", "onco_op", "non_onco_ip", "non_onco_op"]
    classes = []
    if not drugs.empty:
        pv = (drugs.assign(_k=key.values, cost=drugs["cost"].astype(float))
              .groupby(["formulary_class", "formulary_group", "payer_bucket", "_k"])["cost"].sum()
              .unstack("_k").reindex(columns=cells).fillna(0.0).reset_index())
        # Excel's pivot order: class name, then payer
        pv = pv.astype({"formulary_class": str, "payer_bucket": str}).sort_values(
            ["formulary_class", "payer_bucket"], kind="mergesort")
        for r in pv.to_dict("records"):
            onco, non_onco = r["onco_ip"] + r["onco_op"], r["non_onco_ip"] + r["non_onco_op"]
            classes.append({"formulary_class": r["formulary_class"], "formulary_group": str(r["formulary_group"]),
                            "payer": r["payer_bucket"], "onco_ip": _money(r["onco_ip"]),
                            "onco_op": _money(r["onco_op"]), "onco": _money(onco),
                            "non_onco_ip": _money(r["non_onco_ip"]), "non_onco_op": _money(r["non_onco_op"]),
                            "non_onco": _money(non_onco), "total": _money(onco + non_onco)})
    return {"rows": rows, "classes": classes}


def _last(df: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    """Latest period's value of descriptive `cols` per `keys` (names can be re-spelt between
    months' masters; the newest wins)."""
    return (df.sort_values("period").drop_duplicates(keys, keep="last")[keys + cols])


def _round(df: pd.DataFrame, money=(), pct=(), qty=()) -> pd.DataFrame:
    for c in money:
        df[c] = df[c].astype(float).round(2) + 0.0
    for c in pct:
        df[c] = df[c].astype(float).round(4)
    for c in qty:
        df[c] = df[c].astype(float).round(3) + 0.0
    return df


def _page(request: Request, frame: pd.DataFrame, sum_cols: Optional[list] = None, *,
          facets: Sequence[tq.Facet] = (), scope: Sequence[tq.ScopeChip] = ()) -> dict:
    # Same {data,total} paging / sort / search contract as every *-table endpoint; with
    # sum_cols, also {sums} over the filtered rows for the table's totals row. The frame is
    # already aggregated over the page's filters (`scope`, echoed with facets=1); `facets`
    # filter its rows, so sums follow them while share_pct stays a share of the whole selection.
    return da.paginate("formulary", None, request.query_params, col_map={}, frame=frame,
                       sum_cols=sum_cols, facets=facets, scope=scope)


def scope_chips(sc: Scope, *extra: tq.ScopeChip) -> tuple:
    """The page filters a list was aggregated over, as table scope chips, then `extra`."""
    chips = [tq.plant_chip(sc.plant), tq.months_chip(sc.months, _periods())]
    if sc.ptype:
        chips.append(tq.chip("Type", "Patient type", sc.ptype))
    if sc.segment:
        chips.append(tq.chip("Segment", "Segment", sc.segment, SEGMENT_LABELS[sc.segment]))
    if sc.payer:
        chips.append(tq.chip("Payer", "Payer", sc.payer))
    if sc.method != "pooled":
        chips.append(tq.chip("Method", "Method", sc.method, METHODS[sc.method]))
    return (*chips, *extra)


# The items tables' own filters (Month, Hospital and Category come from their columns).
CLASS_FACET = tq.Facet("formulary_class", "Formulary class")
ITEM_FACETS = (CLASS_FACET, tq.Facet("manufacturer", "Manufacturer"))


def _drill_type(sc: Scope) -> None:
    if sc.ptype:
        _bad("The doctor × item table is not split by patient type — drop Type (IP and OP together)")


@router.get("/formulary/doctors")
def formulary_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None),
                      Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                      Payer: str = Query(None), Method: str = Query(None), Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Payer, Method)
    material = _code(Material)
    if material:
        _drill_type(sc)
        return _page(request, _item_doctors_cached(sc, material, _version()), sum_cols=ITEM_SUMS,
                     facets=(CLASS_FACET,), scope=scope_chips(sc, tq.chip("Material", "Item", material)))
    return _page(request, _doctors_cached(sc, _version()), sum_cols=DOCTOR_SUMS, scope=scope_chips(sc))


DOCTOR_SUMS = ["formulary", "out_of_formulary", "non_formulary", "total", "onco", "non_onco", "lines"]
ITEM_SUMS = ["cost", "qty", "lines"]


@_cache
def _doctors_cached(sc: Scope, version: tuple) -> pd.DataFrame:
    d = _cut(_table("formulary_by_doctor"), sc)
    keys = ["plant", "doctor_id"]
    r = _rollup(d, keys)
    r = r.merge(_last(d.astype({k: str for k in keys}), keys, ["doctor_name", "hospital_name"]),
                on=keys, how="left")
    r["compliance_pct"] = r["mean_pct"] if sc.method == "mean" else r["pooled_pct"]
    oof = float(r["out_of_formulary"].sum())
    r["oof_share_pct"] = r["out_of_formulary"] / oof * 100 if oof else np.nan
    # the doctor's whole drug spend (A + B + C) and its onco / non-onco split
    r["total"] = r["formulary"] + r["out_of_formulary"] + r["non_formulary"]
    r = _round(r, money=("formulary", "out_of_formulary", "non_formulary", "total", "onco", "non_onco"),
               pct=("compliance_pct", "oof_share_pct"))
    r = r.sort_values(["out_of_formulary", "non_formulary"], ascending=False, kind="mergesort")
    return r[["doctor_id", "doctor_name", "plant", "hospital_name", "formulary", "out_of_formulary",
              "non_formulary", "compliance_pct", "oof_share_pct", "lines", "total", "onco",
              "non_onco"]].reset_index(drop=True)


@_cache
def _item_doctors_cached(sc: Scope, material: str, version: tuple) -> pd.DataFrame:
    d = _cut(_table("formulary_doctor_item"), sc, ptype=False)
    d = d[d["material"].astype(str) == material].astype({"plant": str, "doctor_id": str})
    keys = ["plant", "doctor_id", "formulary_class"]
    r = d.groupby(keys, as_index=False)[["cost", "qty", "lines"]].sum()
    r = r.merge(_last(d, ["plant", "doctor_id"], ["doctor_name"]), on=["plant", "doctor_id"], how="left")
    hosp = _hospitals()
    r["hospital_name"] = r["plant"].map(lambda p: hosp.get(p, {}).get("his_name"))
    r = _round(r, money=("cost",), qty=("qty",)).sort_values("cost", ascending=False, kind="mergesort")
    return r[["doctor_id", "doctor_name", "plant", "hospital_name", "formulary_class", "cost", "qty",
              "lines"]].reset_index(drop=True)


def _items(d: pd.DataFrame, describe: pd.DataFrame) -> pd.DataFrame:
    """One row per (material, class): sums, hospitals = plants it was billed at, and the
    descriptive columns from `describe` (latest month's master)."""
    d = d.astype({"material": str, "plant": str})
    keys = ["material", "formulary_class", "formulary_group"]
    r = d.groupby(keys, as_index=False).agg(cost=("cost", "sum"), qty=("qty", "sum"),
                                            lines=("lines", "sum"), hospitals=("plant", "nunique"))
    return r.merge(describe, on="material", how="left")


def _describe(by_item: pd.DataFrame) -> pd.DataFrame:
    d = by_item.astype({"material": str})
    return _last(d, ["material"], ["material_desc", "generic_name", "manufacturer", "category"])


@_cache
def _describe_all(version: tuple) -> pd.DataFrame:
    return _describe(_table("formulary_by_item"))


ITEM_COLUMNS = ["material", "material_desc", "generic_name", "manufacturer", "category", "formulary_class",
                "formulary_group", "cost", "qty", "lines", "share_pct", "hospitals"]


@router.get("/formulary/items")
def formulary_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                    Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                    Payer: str = Query(None), Group: str = Query(None), Doctor: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category, Payer)
    group = _choice(Group, {g: g for g in GROUPS}, "Group") or "B"
    doctor = _str(Doctor)
    group_chip = tq.chip("Group", "Formulary group", group, GROUP_LABELS[group])
    if doctor:
        if group == "A":
            _bad("Doctor= lists out-of-formulary (B) or non-formulary (C) items only")
        _drill_type(sc)
        return _page(request, _doctor_items_cached(sc, group, doctor, _version()), sum_cols=ITEM_SUMS,
                     facets=ITEM_FACETS, scope=scope_chips(sc, group_chip, tq.chip("Doctor", "Doctor", doctor)))
    return _page(request, _items_cached(sc, group, _version()), sum_cols=ITEM_SUMS, facets=ITEM_FACETS,
                 scope=scope_chips(sc, group_chip))


@_cache
def _items_cached(sc: Scope, group: str, version: tuple) -> pd.DataFrame:
    d = _cut(_table("formulary_by_item"), sc)
    d = d[d["formulary_group"].astype(str) == group]
    return _finish_items(_items(d, _describe(d)))


@_cache
def _doctor_items_cached(sc: Scope, group: str, doctor: str, version: tuple) -> pd.DataFrame:
    d = _cut(_table("formulary_doctor_item"), sc, ptype=False)
    d = d[(d["doctor_id"].astype(str) == doctor) & (d["formulary_group"].astype(str) == group)]
    # generic name, manufacturer and category are not on the doctor × item grain: take them
    # from the item table, and fall back to the code's first digit for the category.
    describe = _describe_all(version)[["material", "generic_name", "manufacturer", "category"]]
    r = _items(d, describe)
    names = _last(d.astype({"material": str}), ["material"], ["material_desc"])
    r = r.merge(names, on="material", how="left")
    r["category"] = r["category"].fillna(r["material"].str[:1].map(_CATEGORY_OF_DIGIT))
    return _finish_items(r)


def _finish_items(r: pd.DataFrame) -> pd.DataFrame:
    tot = float(r["cost"].sum())
    r["share_pct"] = r["cost"] / tot * 100 if tot else np.nan
    r = _round(r, money=("cost",), pct=("share_pct",), qty=("qty",))
    r = r.sort_values("cost", ascending=False, kind="mergesort")
    return r[ITEM_COLUMNS].reset_index(drop=True)


@router.get("/formulary/scheme/items")
def formulary_scheme_items(request: Request, Plant: str = Query(None), Month: str = Query(None),
                           Type: str = Query(None), Segment: str = Query(None),
                           Category: str = Query(None), Payer: str = Query(None),
                           Kind: str = Query(None)):
    # Payer is validated but not applied: the scheme views are GOV-billed lines by definition.
    sc = _scope(Plant, Month, Type, Segment, Category, Payer)._replace(payer=SCHEME_PAYER)
    kind = _choice(Kind, {k.upper(): k for k in SCHEME_CLASSES}, "Kind") or "restricted"
    return _page(request, _scheme_items_cached(sc, kind, _version()), sum_cols=ITEM_SUMS, facets=ITEM_FACETS,
                 scope=scope_chips(sc, tq.chip("Kind", "Scheme brands", kind, SCHEME_KIND_LABELS[kind])))


@_cache
def _scheme_items_cached(sc: Scope, kind: str, version: tuple) -> pd.DataFrame:
    d = _cut(_table("formulary_by_item"), sc)
    d = d[d["formulary_class"].astype(str).isin(SCHEME_CLASSES[kind])]
    r = _items(d, _describe(d))
    r = _round(r, money=("cost",), qty=("qty",)).sort_values("cost", ascending=False, kind="mergesort")
    return r[["material", "material_desc", "generic_name", "manufacturer", "category", "formulary_class",
              "cost", "qty", "lines", "hospitals"]].reset_index(drop=True)


@router.get("/formulary/validation", dependencies=[Depends(require_admin)])
def formulary_validation():
    """The build's own report: sources read, totals reconciled, agreement with HCG's
    columns, unmatched payers. Admin only — it names payers and per-hospital gaps."""
    v = _validation()
    if v is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return v
