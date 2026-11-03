"""Revenue & Margin — HIS pharmacy sales (IP and OP) at MRP, their cost and the margin between.

Served from app/data/formulary/sales_items.parquet, which `python -m app.etl.formulary` builds
from the same monthly HIS sales lines as the formulary tables: every category (consumables
too), one row per month x unit x patient type x payer x item, with no patient, bill or doctor
column.

    GET /revenue/overview    -> headline and the previous month beside it, IP / OP, HCG's
                                formulary buckets, onco / non-onco / consumables, the unit
                                ranking (each unit with its own buckets, IP and OP beside
                                them) and the monthly trend
    GET /revenue/materials   -> {data,total,sums} one row per item and bucket (facets=1 adds
                                {facets,scope}); the Export CSV pages through it

HCG's sheet method: revenue = total MRP, margin = MRP − cost, margin % = margin ÷ MRP. Every
% is a ratio of SUMS taken after filtering — never an average of percentages across units or
months — and is None when its MRP is 0. A line whose cost price is blank counts as cost 0, as
HCG's sheet does; every block carries how many such lines it holds and their MRP, so the
margin they add is in plain view.

The bucket rule (C1) is shared with the sales export and the executive overview: consumables
are their own bucket whatever their class; a drug is Formulary (A), Out of formulary (B),
Non-formulary (C) by the month's Item Master, or Not in Item Master (X).

/revenue/items and /revenue/insights belong to the older Dec-25 to May-26 page (legacy_kpi.py,
registered first), hence /revenue/materials here.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.api import formulary as F
from app.core import data_access as da
from app.core import table_query as tq

router = APIRouter()

SALES_ITEMS = "sales_items"
NOT_BUILT = ("Sales by item have not been built yet — run python -m app.etl.formulary --from-cache "
             "(seconds, from the line cache)")

ONCO, NON_ONCO, CONSUMABLES = F.ONCO, F.NON_ONCO, F.CONSUMABLES
# Segment keys: the formulary page's two drug segments, plus consumables, which revenue counts.
SEGMENTS = {"onco": ONCO, "non-onco": NON_ONCO, "consumables": CONSUMABLES}
SEGMENT_LABELS = {**F.SEGMENT_LABELS, "consumables": "Consumables"}
_SEGMENT_IN = {**F._SEGMENT_IN, "CONSUMABLES": "consumables", "CONSUMABLE": "consumables"}
TYPE_LABELS = {"IP": "Inpatient (IP)", "OP": "Outpatient (OP)"}

# HCG's revenue buckets, in the order HCG's sheet lists them. Not in Item Master is a drug code
# the month's master does not list: shown only when a selection has some (none so far).
BUCKETS = {"formulary": "Formulary", "out_of_formulary": "Out of formulary",
           "non_formulary": "Non-formulary", "consumables": "Consumables",
           "not_in_master": "Not in Item Master"}
BUCKET_OF_GROUP = {"A": "formulary", "B": "out_of_formulary", "C": "non_formulary",
                   "X": "not_in_master"}
SHOWN_WHEN_NONZERO = ("not_in_master",)

# Summed per block; margin and margin % are derived from the sums, never summed themselves.
MEASURES = ["mrp", "cost", "qty", "lines", "cost_missing_lines", "mrp_cost_missing"]

NOTE_METHOD = ("Revenue = total MRP of the HIS pharmacy sales lines, IP and OP, as HCG's sheet takes it; "
               "margin = MRP − cost price; margin % = margin ÷ MRP. Amounts are summed first and the % "
               "taken once — never an average of percentages.")
NOTE_GST = ("MRP and cost are the HIS file's TOTALMRP and TOTALCOSTPRICE as they stand. The file does "
            "not say whether they include GST, so read margin as MRP less cost, not as profit after "
            "tax.")
NOTE_BUCKETS = ("Formulary, Out of formulary and Non-formulary follow each month's own Item Master; "
                "consumables are a bucket of their own, whatever their class.")
NOTE_NOT_IN_MASTER = "Not in Item Master: drugs the month's Item Master does not list."
NOTE_VIEWS = ("The trend ignores the Month filter and the unit ranking ignores the hospital selector; "
              "every other filter applies to both.")
NOTE_BLANK_COST = ("{lines:,} {noun} in this selection {verb} a blank cost price in the HIS file, with "
                   "MRP ₹{mrp:,.2f}. They count as cost 0, as HCG's sheet does, so that MRP shows in "
                   "full as margin.")
NOTE_NO_BLANK_COST = ("A line with a blank cost price would count as cost 0, as HCG's sheet does; this "
                      "selection has none.")
NOTE_NO_PREVIOUS = "{label} is the first published month, so there is no previous month to compare with."


# ── Data ──────────────────────────────────────────────────────────────────────
# Read through the formulary module's file cache (keyed on mtime/size), from the same folder.
def _path() -> Path:
    return F._data_dir() / f"{SALES_ITEMS}.parquet"


def version() -> tuple:
    """Identity of the published data: the formulary tables, their report and sales_items."""
    return F._version() + (F._stamp(_path()),)


def sales_items() -> pd.DataFrame:
    """The published sales_items table; 503 when it has not been built."""
    p = _path()
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return F._read(str(p), stamp)


def sales_periods() -> list:
    """Months sales_items holds, oldest first (503 when it has not been built)."""
    return _periods_at(version())


@F._cache
def _periods_at(ver: tuple) -> list:
    return sorted(sales_items()["period"].astype(str).unique())


def bucket_of(df: pd.DataFrame) -> pd.Series:
    """C1's bucket key of each sales_items (or formulary_summary) row."""
    b = df["formulary_group"].astype(str).map(BUCKET_OF_GROUP).fillna("not_in_master")
    return b.where(df["category"].astype(str) != CONSUMABLES, "consumables")


# ── Arguments ─────────────────────────────────────────────────────────────────
class Scope(NamedTuple):
    plant: Optional[str]          # SAP plant code; None => all hospitals
    months: tuple                 # periods 'YYYY-MM'
    ptype: Optional[str]          # 'IP' / 'OP'; None => both
    segment: Optional[str]        # 'onco' / 'non-onco' / 'consumables'; None => all
    payer: Optional[str]          # 'CASH' / 'CASH(TPA)' / 'GOV'; None => all


def _scope(Plant, Month, Type, Segment, Payer) -> Scope:
    periods = sales_periods()               # 503 first: nothing below means anything without data
    return Scope(
        plant=F._plant_arg(Plant),
        months=F._months_arg(Month, periods, "sales"),
        ptype=F._choice(Type, {t: t for t in F.TYPES}, "Type"),
        segment=F._choice(Segment, _SEGMENT_IN, "Segment"),
        payer=F._choice(Payer, {p: p for p in F.PAYERS}, "Payer"),
    )


def _cut(df: pd.DataFrame, sc: Scope, *, plant=True, months=True) -> pd.DataFrame:
    """Rows of `df` in scope. The flags drop one filter for the views that ignore it."""
    m = pd.Series(True, index=df.index)
    if plant and sc.plant:
        m &= df["plant"].astype(str) == sc.plant
    if months:
        m &= df["period"].astype(str).isin(sc.months)
    if sc.ptype:
        m &= df["patient_type"].astype(str) == sc.ptype
    if sc.segment:
        m &= df["category"].astype(str) == SEGMENTS[sc.segment]
    if sc.payer:
        m &= df["payer_bucket"].astype(str) == sc.payer
    return df[m]


# ── Arithmetic ────────────────────────────────────────────────────────────────
def _money(v) -> float:
    return round(float(v), 2) + 0.0         # + 0.0 turns -0.0 into 0.0


def _pct(part, whole) -> Optional[float]:
    """part ÷ whole as a % to 2 dp; None when whole is 0 (never NaN, inf or a divide error)."""
    whole = float(whole)
    return round(float(part) / whole * 100, 2) + 0.0 if whole else None


def block(sums) -> dict:
    """The figures every breakdown shares, from summed MEASURES (a Series, dict or row)."""
    mrp, cost = float(sums["mrp"]), float(sums["cost"])
    return {"mrp": _money(mrp), "cost": _money(cost), "margin": _money(mrp - cost),
            "margin_pct": _pct(mrp - cost, mrp), "qty": round(float(sums["qty"]), 3) + 0.0,
            "lines": int(sums["lines"]), "cost_missing_lines": int(sums["cost_missing_lines"]),
            "mrp_cost_missing": _money(sums["mrp_cost_missing"])}


ZERO = dict.fromkeys(MEASURES, 0)


def _sums(df: pd.DataFrame, keys: list) -> dict:
    """MEASURES summed per `keys` value (a tuple when there are several keys)."""
    if df.empty:
        return {}
    g = df.groupby(keys, sort=False)[MEASURES].sum()
    return {k: r for k, r in zip(g.index, g.to_dict("records"))}


def block_of(df: pd.DataFrame) -> dict:
    """Block of a slice of sales_items: the whole slice summed."""
    return block(df[MEASURES].sum() if len(df) else ZERO)


def _split(sums: dict, key, ip_op: dict) -> dict:
    """A key's Block, with its IP and OP Blocks beside it. `key` is one value, or a tuple when
    `sums` was grouped on several keys; `ip_op` is grouped on the same keys + patient type."""
    at = key if isinstance(key, tuple) else (key,)
    return {**block(sums.get(key, ZERO)),
            "ip": block(ip_op.get(at + ("IP",), ZERO)), "op": block(ip_op.get(at + ("OP",), ZERO))}


def _shown_buckets(sums: dict) -> list:
    """C1's bucket keys in HCG's order, from `sums` per bucket: Not in Item Master only when the
    rows hold some of it."""
    return [k for k in BUCKETS if k not in SHOWN_WHEN_NONZERO or sums.get(k, ZERO)["lines"]]


def _buckets(sums: dict, ip_op: dict, whole, keys: list, unit: Optional[str] = None) -> list:
    """One entry per bucket in `keys`: its Block with IP and OP beside it, and its share of
    `whole` (MRP). With `unit`, `sums` is per unit and bucket and the entries are that unit's."""
    at = (lambda k: (unit, k)) if unit is not None else (lambda k: k)
    return [{"key": k, "label": BUCKETS[k], **_split(sums, at(k), ip_op),
             "share_pct": _pct(sums.get(at(k), ZERO)["mrp"], whole)} for k in keys]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/revenue/overview")
def revenue_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                     Segment: str = Query(None), Payer: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Payer)
    return _overview_cached(sc, version())


@F._cache
def _overview_cached(sc: Scope, ver: tuple) -> dict:
    si = sales_items()
    periods = sales_periods()
    names = F._sap_names()
    hosp = F._hospitals()
    cut = _cut(si, sc)
    head = block_of(cut)
    whole = head["mrp"]

    prev_period, prev = None, None
    if len(sc.months) == 1:
        i = periods.index(sc.months[0])
        if i > 0:
            prev_period = periods[i - 1]
            prev = block_of(_cut(si, sc._replace(months=(prev_period,))))

    ty = _sums(cut, ["patient_type"])
    by_type = [{"type": t, "label": TYPE_LABELS[t], **block(ty.get(t, ZERO)),
                "share_pct": _pct(ty.get(t, ZERO)["mrp"], whole)} for t in F.TYPES]

    bk = cut.assign(_bucket=bucket_of(cut))
    bs, bt = _sums(bk, ["_bucket"]), _sums(bk, ["_bucket", "patient_type"])
    by_bucket = _buckets(bs, bt, whole, _shown_buckets(bs))

    cs = _sums(cut, ["category"])
    by_category = [{"key": k, "label": SEGMENT_LABELS[k], **block(cs.get(c, ZERO)),
                    "share_pct": _pct(cs.get(c, ZERO)["mrp"], whole)} for k, c in SEGMENTS.items()]

    # Every unit with sales in the selected months, each on the other filters (zero where the
    # filters leave it nothing). Ranked by MRP; a unit with no MRP has no rank and goes last.
    units = sorted(si.loc[si["period"].astype(str).isin(sc.months), "plant"].astype(str).unique())
    every_unit = _cut(si, sc, plant=False)
    us, ut = _sums(every_unit, ["plant"]), _sums(every_unit, ["plant", "patient_type"])
    # Each unit's buckets, on the same rows as its figures, so they add up to them (HCG's unit x
    # category view). Every unit lists the same buckets, Not in Item Master when any unit has
    # some, so a unit x bucket table lines up.
    eb = every_unit.assign(_bucket=bucket_of(every_unit))
    ub, ubt = _sums(eb, ["plant", "_bucket"]), _sums(eb, ["plant", "_bucket", "patient_type"])
    unit_buckets = _shown_buckets(_sums(eb, ["_bucket"]))
    network = sum(float(us.get(p, ZERO)["mrp"]) for p in units)
    by_hospital = []
    for p in units:
        hh = hosp.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, hh["his_name"] or p),
                            "his_code": hh["his_code"], "his_name": hh["his_name"] or F._his_name(p),
                            **_split(us, p, ut), "share_pct": _pct(us.get(p, ZERO)["mrp"], network),
                            "rank": None, "selected": p == sc.plant,
                            "buckets": _buckets(ub, ubt, us.get(p, ZERO)["mrp"], unit_buckets, unit=p)})
    by_hospital.sort(key=lambda x: (-x["mrp"], x["his_name"] or x["plant"]))
    for i, x in enumerate(b for b in by_hospital if b["mrp"] > 0):
        x["rank"] = i + 1

    over_time = _cut(si, sc, months=False)
    ts, tt = _sums(over_time, ["period"]), _sums(over_time, ["period", "patient_type"])
    trend = [{"period": p, "label": F._month_label(p), **_split(ts, p, tt)} for p in periods]

    notes = [NOTE_METHOD, NOTE_GST, NOTE_BUCKETS]
    if any(b["key"] == "not_in_master" for b in by_bucket):
        notes.append(NOTE_NOT_IN_MASTER)
    notes.append(NOTE_VIEWS)
    n = head["cost_missing_lines"]
    notes.append(NOTE_BLANK_COST.format(lines=n, noun="line" if n == 1 else "lines",
                                        verb="has" if n == 1 else "have", mrp=head["mrp_cost_missing"])
                 if n else NOTE_NO_BLANK_COST)
    if len(sc.months) > 1:
        notes.append(F.NOTE_MONTHS)
    if len(sc.months) == 1 and prev_period is None:
        notes.append(NOTE_NO_PREVIOUS.format(label=F._month_label(sc.months[0])))
    if sc.plant and cut.empty:
        notes.append(f"No pharmacy sales for {names.get(sc.plant, sc.plant)} in this selection.")

    return {
        "scope": {"plant": sc.plant or "ALL",
                  "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
                  "his_name": F._his_name(sc.plant) if sc.plant else None, "months": list(sc.months),
                  "type": sc.ptype or "ALL", "segment": sc.segment or "ALL", "payer": sc.payer or "ALL"},
        "months_available": periods,
        "month_labels": {p: F._month_label(p) for p in periods},
        "headline": head,
        "prev_period": prev_period,
        "prev": prev,
        "by_type": by_type,
        "by_bucket": by_bucket,
        "by_category": by_category,
        "by_hospital": by_hospital,
        "trend": trend,
        "notes": notes,
        "data_as_of": F._built_at(),
    }


def scope_chips(sc: Scope) -> tuple:
    """The page filters a list was aggregated over, as table scope chips."""
    chips = [tq.plant_chip(sc.plant), tq.months_chip(sc.months, sales_periods())]
    if sc.ptype:
        chips.append(tq.chip("Type", "Patient type", sc.ptype))
    if sc.segment:
        chips.append(tq.chip("Segment", "Segment", sc.segment, SEGMENT_LABELS[sc.segment]))
    if sc.payer:
        chips.append(tq.chip("Payer", "Payer", sc.payer))
    return tuple(chips)


MATERIAL_FACETS = (tq.Facet("category", "Category"), tq.Facet("bucket_label", "Formulary bucket"),
                   tq.Facet("manufacturer", "Manufacturer"))
MATERIAL_SUMS = ["mrp", "cost", "margin", "qty", "lines", "cost_missing_lines"]
MATERIAL_COLUMNS = ["material", "material_desc", "generic_name", "manufacturer", "category", "bucket",
                    "bucket_label", "mrp", "cost", "margin", "margin_pct", "qty", "lines", "hospitals",
                    "cost_missing_lines"]


@router.get("/revenue/materials")
def revenue_materials(request: Request, Plant: str = Query(None), Month: str = Query(None),
                      Type: str = Query(None), Segment: str = Query(None), Payer: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Payer)
    return da.paginate(SALES_ITEMS, None, request.query_params, col_map={},
                       frame=_materials_cached(sc, version()), sum_cols=MATERIAL_SUMS,
                       facets=MATERIAL_FACETS, scope=scope_chips(sc), max_page_size=tq.EXPORT_MAX_PAGE)


@F._cache
def _materials_cached(sc: Scope, ver: tuple) -> pd.DataFrame:
    """One row per item and bucket over the scope (an item whose class changed between the
    selected months' masters has a row per bucket, so the buckets add up to the overview's).
    Names come from the latest month in the selection."""
    d = _cut(sales_items(), sc)
    d = d.assign(bucket=bucket_of(d), material=d["material"].astype(str), plant=d["plant"].astype(str))
    r = d.groupby(["material", "bucket"], as_index=False).agg(
        mrp=("mrp", "sum"), cost=("cost", "sum"), qty=("qty", "sum"), lines=("lines", "sum"),
        cost_missing_lines=("cost_missing_lines", "sum"), hospitals=("plant", "nunique"))
    describe = (d.sort_values("period").drop_duplicates("material", keep="last")
                [["material", "material_desc", "generic_name", "manufacturer", "category"]])
    r = r.merge(describe, on="material", how="left")
    r["bucket_label"] = r["bucket"].map(BUCKETS)
    r["margin"] = r["mrp"] - r["cost"]
    r["margin_pct"] = (r["margin"] / r["mrp"].where(r["mrp"] != 0) * 100).round(2) + 0.0
    for c in ("mrp", "cost", "margin"):
        r[c] = r[c].astype(float).round(2) + 0.0
    r["qty"] = r["qty"].astype(float).round(3) + 0.0
    r = r.sort_values(["mrp", "material"], ascending=[False, True], kind="mergesort")
    return r[MATERIAL_COLUMNS].reset_index(drop=True)
