"""Generic, registry-driven KPI API.

A single pair of endpoints serves every aggregate parquet, replacing ~20 boilerplate
api/service file pairs. Each KPI is described once in REGISTRY.

    GET /kpi/{key}            -> chart records (filter by Plant + Material|MaterialGroup)
    GET /kpi/{key}/table      -> paginated {data,total}
    GET /meta/kpis            -> registry listing
    GET /meta/plants          -> hospital/plant options
    GET /meta/materials       -> material catalogue (optionally by plant)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from app.core import data_access as da
from app.core import drill_sources as ds

router = APIRouter()

# key -> (parquet table, status). status: available | proxy
REGISTRY = {
    # Inventory
    "current-stock-value": ("kpi_stock_value", "available"),
    "inventory-aging": ("kpi_inventory_aging", "available"),
    "inventory-turnover-ratio": ("kpi_health_score", "proxy"),
    "aging-distribution": ("kpi_aging_distribution", "available"),
    "days-on-hand": ("kpi_doh", "available"),
    "inventory-health-score": ("kpi_health_score", "available"),
    "non-moving-inventory": ("kpi_non_moving", "available"),
    "inventory-risk": ("kpi_risk_classification", "available"),
    "stock-change": ("kpi_stock_change", "available"),
    # Procurement
    "purchase-value": ("kpi_purchase_value", "available"),
    "monthly-purchase-value": ("kpi_monthly_purchase_value", "available"),
    "procurement-variance": ("kpi_procurement_variance", "available"),
    "vendor-volume-contribution": ("kpi_vendor_volume", "available"),
    "purchase-by-location": ("kpi_purchase_by_location", "available"),
    # Consumption
    "unit-sold-per-sku": ("kpi_units_consumed", "available"),
    "consumption-by-department": ("kpi_consumption_by_department", "available"),
    # Forecasting (D3/D5/D6 derived tables; D1/D4/D8 via /inventory/replenishment-data)
    "fulfillment-rate": ("kpi_fulfillment", "available"),
    "stock-radar": ("kpi_stock_radar", "available"),
    "aging-risk-forecast": ("kpi_aging_risk_forecast", "available"),
    # Additional
    "near-expiry": ("kpi_near_expiry", "available"),
    "procurement-cycle-time": ("kpi_cycle_time", "available"),
    "vendor-lead-time": ("kpi_vendor_lead_time", "available"),
    "fill-rate": ("kpi_fill_rate", "available"),
    "forecast-accuracy": ("forecast_accuracy", "proxy"),
}


def _resolve(key: str) -> str:
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown KPI key '{key}'")
    return REGISTRY[key][0]


# Static-snapshot chart/summary results are pure functions of their args → memoize so
# every KPI detail page is instant after the first hit. Table pagination stays uncached
# (its search/sort/filter params vary too widely to cache usefully).
_KPI_RESULT_CACHES: list = []


def _kc(fn):
    cached = lru_cache(maxsize=256)(fn)
    _KPI_RESULT_CACHES.append(cached)
    return cached


def clear_kpi_caches() -> None:
    for c in _KPI_RESULT_CACHES:
        c.cache_clear()
    _categories_cached.cache_clear()
    _matrix_for.cache_clear()
    ds.clear_caches()


# Tables that were pre-aggregated PAST material grain, but whose exact parent frame
# drill_sources already rebuilds. Without this, `?Category=` on kpi_aging_distribution
# was silently inert: the parquet is plant x material_group x aging_bucket, so there is
# no category column for filter_category to touch and the response came back identical
# under every category — an invisible no-op, which is the exact failure mode this whole
# change set exists to remove.
#
# The substitution happens ONLY when a category actually resolves. With no category the
# request runs down the original code path against the original parquet, so every
# unfiltered number is not merely equal but produced by identical code. And the
# substitute is not a lookalike: drill_inventory_grain collapsed onto
# (plant, material_group, aging_bucket) reproduces kpi_aging_distribution cell for cell
# on stock_value, stock_qty AND sku_count — asserted in tests/test_drilldown.py.
_CATEGORY_REGRAIN = {"kpi_aging_distribution": "drill_inventory_grain"}


def _regrained(table: str, category):
    """The frame to serve `table` from, or None to use the table itself.

    Two substitution routes, both inert without a category:

      * _CATEGORY_REGRAIN — swap in the richer-grain frame WHOLE and let
        da.filter_category do the cut (inventory; the frame's `category` IS the
        material category, so filtering it is correct).
      * ds.PROC_REGRAIN — the procurement aggregates, where the same trick would be
        actively wrong: fact_po's `category` is the PO SPEND taxonomy, so handing the
        raw grain to filter_category would either match nothing or (as today) be
        refused by _is_derived_category_col and silently return the whole portfolio.
        These builders therefore cut on `material_category` themselves and hand back
        a frame already collapsed to the aggregate's own grain — one that carries no
        derived category column at all, so the filter_category call downstream is a
        guaranteed no-op rather than a second, conflicting filter.
    """
    src = _CATEGORY_REGRAIN.get(table)
    if src and da.resolve_category(category):
        return ds.load_source(src, None)
    return ds.regrain(table, da.resolve_category(category))


def _category_unsupported(table: str, category) -> None:
    """400 rather than silently ignore a Category the table cannot honour.

    Only fires when a caller EXPLICITLY passes a bucket for a KPI listed in
    ds.CATEGORY_UNSUPPORTED. Returning the unfiltered portfolio under an "Onco Drugs"
    label is the one outcome worse than refusing: the reader cannot tell a filter that
    did nothing from a filter that found everything. GET /meta/category-support
    publishes the same list so a card can decide not to offer the control at all.
    """
    if da.resolve_category(category) and table in ds.CATEGORY_UNSUPPORTED:
        raise HTTPException(status_code=400, detail=ds.CATEGORY_UNSUPPORTED[table])


@_kc
def _kpi_chart_cached(table, plant, material, material_group, group_by, measures, top, category=None):
    return da.chart_series(table, plant=plant, material=material, material_group=material_group,
                           group_by=group_by, measures=measures, top=top, category=category,
                           frame=_regrained(table, category))


@_kc
def _kpi_summary_cached(table, plant, material, material_group, category=None):
    return da.summarize(table, plant=plant, material=material, material_group=material_group,
                        category=category, frame=_regrained(table, category))


# `Category` is OPTIONAL everywhere and defaults to None => filter_category is a
# no-op, so every existing call site keeps its exact current response. On tables
# with no material grain no `category` column is derived at all — see
# data_access._attach_category — so the param used to be inert there. It is no longer:
# _CATEGORY_REGRAIN routes the inventory aggregate to the fact-grain frame it was built
# from, and ds.PROC_REGRAIN rebuilds six of the eight procurement aggregates from
# fact_po / fact_grn at material grain. The one procurement KPI that genuinely cannot
# take the cut now says so with a 400 instead of quietly answering unfiltered.
@router.get("/kpi/{key}")
def kpi_chart(
    key: str,
    plant: Optional[str] = Query(None, alias="Plant"),
    material: Optional[str] = Query(None, alias="Material"),
    material_group: Optional[str] = Query(None, alias="MaterialGroup"),
    category: Optional[str] = Query(None, alias="Category"),
    group_by: Optional[str] = Query(None, description="comma cols to aggregate by"),
    measures: Optional[str] = Query(None, description="comma numeric cols to sum"),
    top: Optional[int] = Query(None, description="keep top-N rows by first measure"),
):
    table = _resolve(key)
    _category_unsupported(table, category)
    return _kpi_chart_cached(table, plant, material, material_group, group_by,
                             measures, top, category)


@router.get("/kpi/{key}/summary")
def kpi_summary(
    key: str,
    plant: Optional[str] = Query(None, alias="Plant"),
    material: Optional[str] = Query(None, alias="Material"),
    material_group: Optional[str] = Query(None, alias="MaterialGroup"),
    category: Optional[str] = Query(None, alias="Category"),
):
    table = _resolve(key)
    _category_unsupported(table, category)
    return _kpi_summary_cached(table, plant, material, material_group, category)


@router.get("/kpi/{key}/table")
def kpi_table(
    key: str,
    request: Request,
    plant: Optional[str] = Query(None, alias="Plant"),
    category: Optional[str] = Query(None, alias="Category"),
):
    table = _resolve(key)
    _category_unsupported(table, category)
    params = dict(request.query_params)
    return da.paginate(table, plant, params, col_map={}, category=category,
                       frame=_regrained(table, category))


PORTFOLIOS = {
    "inventory": ["current-stock-value", "inventory-aging", "days-on-hand", "stock-change",
                  "aging-distribution", "inventory-health-score", "non-moving-inventory",
                  "inventory-risk", "near-expiry"],
    "procurement": ["purchase-value", "monthly-purchase-value", "procurement-variance",
                    "vendor-volume-contribution", "purchase-by-location", "procurement-cycle-time",
                    "vendor-lead-time", "fill-rate"],
    "consumption": ["unit-sold-per-sku", "consumption-by-department"],
    "forecasting": ["fulfillment-rate", "stock-radar", "aging-risk-forecast"],
}


@router.get("/portfolio/{name}/summary")
def portfolio_summary(name: str, plant: Optional[str] = Query(None, alias="Plant"),
                      category: Optional[str] = Query(None, alias="Category")):
    if name not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio '{name}'")
    return _portfolio_summary_cached(name, da.resolve_plant(plant), da.resolve_category(category))


# Static snapshot → the summary for (portfolio, plant, category) never changes until refresh.
@lru_cache(maxsize=256)
def _portfolio_summary_cached(name: str, plant: Optional[str], category: Optional[str] = None):
    return {k: da.summarize(REGISTRY[k][0], plant=plant, category=category,
                            frame=_regrained(REGISTRY[k][0], category))
            for k in PORTFOLIOS[name]}


def warmup() -> None:
    """Precompute every portfolio's All-Plants summary at startup."""
    for name in PORTFOLIOS:
        try:
            _portfolio_summary_cached(name, None)
        except Exception:
            pass


@router.get("/meta/kpis")
def meta_kpis():
    return [{"key": k, "table": t, "status": s} for k, (t, s) in REGISTRY.items()]


@lru_cache(maxsize=1)
def _plant_domains():
    """Which data domains each plant actually has data for — so the UI can hide
    plants that would only show zeros on a given section (e.g. corporate offices &
    labs have no inventory)."""
    inv = set(da.load("fact_inventory")["plant"].astype(str))
    con = set(da.load("fact_consumption")["plant"].astype(str))
    grn = set(da.load("fact_grn")["plant"].astype(str))
    return inv, con, grn


@router.get("/meta/plants")
def meta_plants():
    df = da.load("dim_plant")
    inv, con, grn = _plant_domains()
    items = []
    for r in df.to_dict("records"):
        code = str(r["plant"])
        domains = []
        if code in inv:
            domains.append("inventory")
        if code in con:
            domains += ["consumption", "forecasting"]   # forecasts are built from consumption
        if code in grn:
            domains.append("procurement")
        items.append({"code": r["plant"], "name": r.get("plant_name", r["plant"]), "domains": domains})
    items = [it for it in items if it["domains"]]        # drop plants with no data at all
    items.sort(key=lambda x: x["name"])
    return {"plants": [{"code": "ALL", "name": "All Plants",
                        "domains": ["inventory", "consumption", "forecasting", "procurement"]}] + items}


@router.get("/meta/materials")
def meta_materials(plant: Optional[str] = Query(None, alias="Plant")):
    df = da.load("dim_material")[["material", "material_desc", "material_group"]].copy()
    if plant and plant.upper() != "ALL":
        sv = da.filter_plant(da.load("kpi_stock_value"), plant)
        df = df[df["material"].isin(sv["material"].unique())]
    return {"materials": da._clean_records(df)}


@router.get("/meta/material-groups")
def meta_material_groups(plant: Optional[str] = Query(None, alias="Plant")):
    df = da.load("dim_material")
    groups = sorted(g for g in df["material_group"].dropna().unique() if str(g) not in ("nan", ""))
    return {"groups": groups}


@router.get("/meta/vendors")
def meta_vendors(plant: Optional[str] = Query(None, alias="Plant")):
    df = da.load("kpi_vendor_volume")
    df = da.filter_plant(df, plant)
    vendors = sorted(v for v in df["vendor_name"].dropna().unique() if str(v) not in ("nan", ""))
    return {"vendors": vendors}


@router.get("/meta/category-support")
def meta_category_support():
    """Which KPIs actually honour `?Category=`, and how — the contract a card wires off.

    A card-level category control is only honest if the endpoint behind it really
    narrows. Three outcomes exist and they are NOT distinguishable from the response
    alone, which is exactly how a dead control survives review:

      supported=true,  how="native"   the aggregate carries a material category
                                      column, filter_category cuts it directly.
      supported=true,  how="regrain"  the aggregate was built past material grain, so
                                      the request is rebuilt from fact_po / fact_grn at
                                      material grain (`source` names the fact frame).
      supported=false                 the cut would be misleading; `reason` says why,
                                      and passing a Category returns 400 rather than
                                      the unfiltered portfolio under a filtered label.

    Computed from the same dicts the request path uses (ds.PROC_REGRAIN,
    _CATEGORY_REGRAIN, ds.CATEGORY_UNSUPPORTED) plus a live column check, so it cannot
    drift from what the endpoints will actually do.
    """
    out = []
    for key, (table, _status) in REGISTRY.items():
        if table in ds.CATEGORY_UNSUPPORTED:
            out.append({"key": key, "table": table, "supported": False, "how": None,
                        "source": None, "reason": ds.CATEGORY_UNSUPPORTED[table]})
            continue
        if table in ds.PROC_REGRAIN:
            src = "fact_grn" if table == "kpi_cycle_time" else "fact_po"
            out.append({"key": key, "table": table, "supported": True, "how": "regrain",
                        "source": src, "reason": None})
            continue
        if table in _CATEGORY_REGRAIN:
            out.append({"key": key, "table": table, "supported": True, "how": "regrain",
                        "source": _CATEGORY_REGRAIN[table], "reason": None})
            continue
        try:
            df = da.load(table)
            native = "category" in df.columns and da._is_derived_category_col(df)
        except Exception:
            native = False
        out.append({"key": key, "table": table, "supported": bool(native),
                    "how": "native" if native else None, "source": table if native else None,
                    "reason": None if native else
                    f"{table} carries no material grain and has no fact-grain rebuild."})
    return {"categories": da.CATEGORIES, "kpis": sorted(out, key=lambda r: r["key"])}


@router.get("/meta/categories")
def meta_categories(plant: Optional[str] = Query(None, alias="Plant")):
    """Options for the global Category selector (sibling of /meta/plants).

    "All" first (the default — no filtering), then the buckets in a fixed display
    order with their live share of stock value. `Unclassified` is returned ONLY when
    it actually holds stock, so the selector never offers an empty bucket — but it is
    never hidden when it does hold value.

    Each bucket also carries a `coverage` block, because the source data is NOT
    uniform across domains and a silently-empty chart would read as a broken filter:
    HCG dispenses onco drugs through the IP/OP BILLING path (Rs 297 Cr of billed
    revenue), not through internal consumption movements (3 rows, Rs 1.8k in
    fact_consumption over the whole 6-month window). So selecting "Onco Drugs" on any
    consumption-derived page — units consumed, days-on-hand, demand & cash-flow
    forecast, replenishment quantities — legitimately shows almost nothing. The flags
    below let the UI say "this category has no consumption data" instead of drawing
    an empty chart and letting the reader conclude the filter is broken.
    """
    return _categories_cached(da.resolve_plant(plant))


@lru_cache(maxsize=64)
def _categories_cached(plant: Optional[str]):
    sv = da.filter_plant(da.load("kpi_stock_value"), plant)
    uc = da.filter_plant(da.load("kpi_units_consumed"), plant)
    try:                       # billing aggregates have no plant dimension
        bill = da.load("sales_by_material").groupby("category", observed=True)["revenue"].sum()
    except Exception:
        bill = None

    total = float(sv["stock_value_cost"].sum())
    by = sv.groupby("category", observed=True).agg(
        value=("stock_value_cost", "sum"), skus=("material", "nunique")).to_dict("index")
    cons = uc.groupby("category", observed=True)["consumption_cost"].sum()
    # Reorder lines per category — the third domain a bucket can be non-empty in, and
    # the one where hiding a bucket does the most damage (see the guard below).
    try:
        _rp = da.filter_plant(da.load("stock_replenishment_and_aging_risk"), plant)
        _rq = pd.to_numeric(_rp["replenishment_quantity"], errors="coerce").fillna(0.0)
        reorder = _rp[_rq > 0].groupby("category", observed=True).size()
    except Exception:
        reorder = None

    items = []
    for name in da.CATEGORIES:
        row = by.get(name)
        value = float(row["value"]) if row else 0.0
        skus = int(row["skus"]) if row else 0
        cval = float(cons.get(name, 0.0))
        rlines = int(reorder.get(name, 0)) if reorder is not None else 0
        # A bucket is only hidden when it is empty in EVERY domain — not merely empty
        # on stock. Unclassified holds no stock but ~50% of consumption cost and 10,614
        # of the 19,014 reorder lines; hiding it on the stock test alone meant picking
        # any category silently dropped over half the reorder queue, on the very feature
        # built to stop lines from going missing. The offered buckets must always sum
        # back to "All" on every domain they can be applied to.
        if name == da.CATEGORY_UNCLASSIFIED and skus == 0 and cval <= 0 and rlines == 0:
            continue
        bval = float(bill.get(name, 0.0)) if bill is not None else 0.0
        # "has consumption data" is judged against this category's own stock value:
        # a bucket holding Rs 25 Cr of stock and Rs 1.8k of consumption has, for any
        # practical purpose, no consumption history at all.
        has_cons = bool(value <= 0 or cval / value > 0.01)
        items.append({"key": name, "name": name, "value": value, "skus": skus,
                      "share_pct": round(value / total * 100, 2) if total else 0.0,
                      "coverage": {"stock_value": value, "consumption_cost": cval,
                                   "billed_revenue": bval, "reorder_lines": rlines,
                                   "has_stock": value > 0, "has_consumption": has_cons,
                                   "has_billing": bval > 0, "has_reorder": rlines > 0,
                                   "note": (None if has_cons else
                                            "Dispensed via IP/OP billing, not internal "
                                            "consumption — consumption-derived KPIs "
                                            "(units consumed, days-on-hand, demand & "
                                            "cash-flow forecast, replenishment) have "
                                            "little or no data for this category.")}})
    return {"categories": [{"key": "All", "name": "All Categories", "value": total,
                            "skus": int(sv["material"].nunique()),
                            "share_pct": 100.0 if total else 0.0}] + items,
            "total_value": total}


# ============================ HOVER DRILL-DOWN ===============================
# ONE generic endpoint behind every "hover a bar/slice -> show me the top 10 items
# inside it" interaction, instead of a bespoke endpoint per chart. Given a KPI, the
# dimension the chart is grouped by, and the specific slice the user hovered, it
# returns the top-N underlying items with name + value + share. Reuses the exact
# same filtering conventions as the rest of this file (da.filter_plant /
# da.filter_category / REGISTRY), so a drill-down can never disagree with the chart
# it was launched from.
#
# A KPI is NOT pinned to one table. Many aggregates are pre-aggregated past the grain
# the drill needs — kpi_aging_distribution has no material column, kpi_vendor_volume
# has no material column, kpi_fill_rate is one row per plant — so the request is
# resolved against an ORDERED CHAIN of sources: the KPI's own table first (every
# combination that already worked keeps its exact numbers and its exact table), then
# the fact-grain frame the ETL built that table FROM (app/core/drill_sources.py).
# Rejecting the request is the last resort, not the first response, because a hover
# that silently does nothing is indistinguishable from a chart nobody wired up.

# dim alias -> column(s) on the frame. First present column wins.
_DIM_DIRECT = {
    "material": ("material", "material_id"),
    "material_group": ("material_group",),
    "plant": ("plant",),
    "category": ("category",),
    # Strict on purpose. `category` means the PO SPEND taxonomy on the procurement
    # tables and the derived MATERIAL category everywhere else, so these two aliases
    # must never silently resolve to a `category` column of the other vocabulary —
    # they only ever match the explicitly-named columns the richer-grain frames
    # carry. Use `dim=category` for whatever the parent chart itself means by it.
    "po_category": ("po_category",),
    "material_category": ("material_category",),
    "vendor": ("vendor_name",),
    "month": ("month",),
    "year": ("year",),
    "department": ("cost_ctr",),
    "batch": ("batch",),
    "sloc": ("sloc",),
    "expiry_bucket": ("expiry_bucket",),
    "aging_category": ("aging_category",),
    "aging_bucket": ("aging_bucket",),
    # The Days-on-Hand coverage histogram's own ladder. Not derivable from `doh_days`
    # here the way the aging bands are, because "no movement" is a seventh band that
    # means `doh_days IS NULL` rather than a range — so drill_doh_grain materialises
    # the column with the endpoint's own definition and this just reads it.
    "doh_band": ("doh_band",),
    "aging_risk": ("aging_risk",),
    "risk_level": ("risk_level",),
    "health_tier": ("health_tier",),
    "radar_status": ("radar_status",),
    "reason": ("reason",),
    "aging_risk_forecast": ("aging_risk_forecast",),
    "will_stagnate": ("will_stagnate",),
    "consumed": ("consumed",),
    "priority_band": ("priority_band",),
    "priority_label": ("priority_label",),
    "status": ("status",),
    "metric": ("metric",),
}

# What a row with no key is called. Never "nan", never silently dropped.
UNASSIGNED = "(unassigned)"

# Derived aging band — the ladder every aging chart in the UI uses. Identical to the
# bins kpi_aging_distribution and kpi_inventory_aging were cut with.
_AGING_BANDS = [(-1, 30, "0-30"), (30, 90, "31-90"), (90, 180, "91-180"),
                (180, 365, "181-365"), (365, float("inf"), "365+")]

# by-alias -> (key column candidates, label column candidates | None)
_BY_DIRECT = {
    "material": (("material", "material_id"), ("material_desc", "desc")),
    "vendor": (("vendor_name",), None),
    "plant": (("plant",), None),
    "material_group": (("material_group",), None),
    "category": (("category",), None),
    "department": (("cost_ctr",), ("department_name",)),
    "batch": (("batch",), None),
}

# Order the endpoint degrades through when the requested `by` exists on no source.
# Only genuine ENTITY grains: falling back from "top materials" to "top months" would
# answer a different question while looking like an answer to this one.
_BY_FALLBACK_ORDER = ["material", "vendor", "material_group", "department", "batch",
                      "plant", "category"]

# Default measure per table — the number the chart itself is showing.
_DEFAULT_MEASURE = {
    "kpi_stock_value": "stock_value_cost",
    "kpi_inventory_aging": "closing_stock_value",
    "kpi_aging_distribution": "stock_value",
    "kpi_doh": "stock_qty",
    "kpi_health_score": "closing_stock_value",
    "kpi_non_moving": "closing_stock_value",
    "kpi_risk_classification": "closing_stock_value",
    "kpi_stock_change": "stock_change",
    "kpi_near_expiry": "total_cost",
    "kpi_units_consumed": "total_units",
    "kpi_consumption_by_department": "consumption_cost",
    "kpi_purchase_value": "purchase_value",
    "kpi_monthly_purchase_value": "monthly_purchase_value",
    "kpi_vendor_volume": "vendor_value",
    "kpi_purchase_by_location": "purchase_value",
    "kpi_stock_radar": "closing_stock",
    "kpi_aging_risk_forecast": "closing_stock",
    "kpi_fulfillment": "closing_stock",
    "kpi_cycle_time": "gr_lines",
    "kpi_vendor_lead_time": "gr_lines",
    "kpi_fill_rate": "ordered_qty",
    "kpi_procurement_variance": "purchase_value",
    "stock_replenishment_and_aging_risk": "replenishment_quantity",
    "drill_po_grain": "purchase_value",
    "drill_grn_grain": "grn_value",
    "drill_inventory_grain": "stock_value",
    "drill_consumption_grain": "consumption_cost",
    "drill_doh_grain": "stock_value_cost",
    "drill_expired_grain": "expired_value",
    "drill_replen_priority": "replenishment_quantity",
}

# NON-ADDITIVE measures. Summing a ratio, a score or an average across children is
# arithmetically false — "the top 10 vendors inside this 12-day average lead time"
# adding up to 118 days is a confidently wrong answer. Each of these instead
# aggregates as a WEIGHTED MEAN, sum(m*w)/sum(w), over the first weight column
# present. That is not a cosmetic choice: for a ratio whose denominator IS the weight
# (lead time weighted by lines, fill rate by ordered qty, turnover by stock value,
# DOH by daily consumption, coverage by monthly demand) the weighted mean is EXACTLY
# the aggregate ratio, not an approximation of it.
# Responses on these measures report additive=false and null shares, because a
# percentage of an average is meaningless and a tooltip must never imply otherwise.
_WEIGHTED_MEASURE = {
    "avg_lead_time_days": ("po_tat_n", "gr_lines"),
    "median_lead_time_days": ("po_tat_n", "gr_lines"),
    "avg_po_to_gr_tat": ("po_tat_n", "gr_lines"),
    "avg_pr_to_gr_tat": ("pr_tat_n", "gr_lines"),
    "fill_rate_pct": ("ordered_qty",),
    "doh_days": ("avg_daily_consumption",),
    "turnover_annualized": ("closing_stock_value",),
    "health_score": ("closing_stock_value",),
    "aging_days": ("closing_stock_value", "stock_value", "stock_qty", "closing_stock"),
    "projected_aging_days": ("closing_stock",),
    "coverage_months": ("demand_monthly", "demand_forecast"),
    "fulfillment_rate": ("demand_monthly",),
    "variance_pct": ("prev_value",),
    "value_share_pct": ("vendor_value",),
    "unit_cost": ("closing_stock",),
    "moving_avg_price": ("stock_qty",),
    "days_to_expiry": ("total_cost", "closing_stock_value"),
    "cover": ("demand_monthly",),
    "lead_months": ("demand_monthly",),
}

# Measures whose weighted mean is only an APPROXIMATION of the true aggregate, and
# which therefore say so in the response instead of quietly pretending otherwise.
_APPROXIMATE_MEASURE = {"median_lead_time_days", "fulfillment_rate", "health_score",
                        "value_share_pct", "moving_avg_price"}

# Extra drill targets that are not registry KPI keys.
_EXTRA_TABLES = {"reorder-priority": "stock_replenishment_and_aging_risk",
                 "replenishment": "stock_replenishment_and_aging_risk",
                 # Wastage % has no registry KPI because it is a RATIO of two tables.
                 # Its numerator is not: drill_expired_grain is the expired lines the
                 # rupee figure is summed from, so a plant bar can list the items that
                 # actually expired there. Ranked by expired_value, never by the ratio.
                 "wastage-rate": "drill_expired_grain",
                 "inventory": "fact_inventory"}

# Keys the frontend uses on pages whose chart is built from a differently-named KPI.
_KPI_ALIAS = {
    "inventory-valuation": "current-stock-value",   # the Valuation page charts cost vs MRP
    "stock-valuation": "current-stock-value",
    "inventory-turnover": "inventory-turnover-ratio",
    "consumption-by-sku": "unit-sold-per-sku",
    "units-consumed": "unit-sold-per-sku",
    "vendor-contribution": "vendor-volume-contribution",
    "reorder-priority-bands": "reorder-priority",
}

# kpi -> the richer-grain sources to try after its own table. Ordered; the first
# source that can answer (dim AND by AND measure) wins, so a KPI whose own table is
# sufficient never leaves it and never changes its numbers.
_FALLBACK_SOURCES = {
    "current-stock-value": ["drill_inventory_grain"],
    "inventory-aging": ["drill_inventory_grain"],
    "inventory-turnover-ratio": ["drill_inventory_grain"],
    "inventory-health-score": ["drill_inventory_grain"],
    "aging-distribution": ["drill_inventory_grain"],
    "non-moving-inventory": ["drill_inventory_grain"],
    "inventory-risk": ["drill_inventory_grain"],
    # drill_doh_grain is LAST on purpose: every combination kpi_doh or
    # drill_inventory_grain already answered keeps its exact source and its exact
    # numbers, and only dim=doh_band / measure=stock_value_cost — which neither of
    # them carries — falls through to it.
    "days-on-hand": ["drill_inventory_grain", "drill_doh_grain"],
    "near-expiry": ["drill_inventory_grain"],
    "stock-change": ["drill_consumption_grain", "drill_grn_grain"],
    "purchase-value": ["drill_po_grain"],
    "monthly-purchase-value": ["drill_po_grain"],
    "procurement-variance": ["drill_po_grain"],
    "vendor-volume-contribution": ["drill_po_grain"],
    "purchase-by-location": ["drill_po_grain"],
    "fill-rate": ["drill_po_grain"],
    "procurement-cycle-time": ["drill_grn_grain"],
    "vendor-lead-time": ["drill_grn_grain"],
    "unit-sold-per-sku": ["drill_consumption_grain"],
    "consumption-by-department": ["drill_consumption_grain"],
    "reorder-priority": ["drill_replen_priority"],
    "replenishment": ["drill_replen_priority"],
}


def _drill_kpi(kpi: str) -> str:
    return _KPI_ALIAS.get(kpi, kpi)


def _drill_table(kpi: str) -> str:
    if kpi in REGISTRY:
        return REGISTRY[kpi][0]
    if kpi in _EXTRA_TABLES:
        return _EXTRA_TABLES[kpi]
    raise HTTPException(status_code=404, detail=f"Unknown KPI/table '{kpi}' for drill-down")


def _sources_for(kpi: str) -> list:
    return [_drill_table(kpi)] + list(_FALLBACK_SOURCES.get(kpi, []))


def _first_col(df, names):
    if not names:
        return None
    for n in names:
        if n in df.columns:
            return n
    return None


# ── dimension / by / measure resolution on ONE candidate frame ────────────────

def _dim_plan(df, dim: str):
    """(kind, column, resolved_name) for `dim` on this frame, or None if absent."""
    if dim == "year_month":
        # `month` alone is NOT a period on any monthly table here — kpi_stock_change
        # holds December 2025 AND December 2026, so dim=month, slice="December"
        # answers with both months summed while the user is pointing at one bar. The
        # panel would then print a number visibly larger than the bar and look
        # authoritative doing it. A period needs the year in the key, so this is the
        # dimension every month-grouped chart binds.
        if "year" in df.columns and "month" in df.columns:
            return ("derived_year_month", None, "year_month")
        return None
    if dim in ("aging_band", "aging"):
        if "aging_bucket" in df.columns:      # already cut with the same ladder
            return ("direct", "aging_bucket", "aging_band")
        if "aging_days" in df.columns:
            return ("derived_aging", "aging_days", "aging_band")
        return None
    col = _first_col(df, _DIM_DIRECT.get(dim, (dim,)))
    return ("direct", col, col) if col else None


def _keys(series):
    """A column rendered as the STRINGS this endpoint speaks in.

    Every key it returns must be passable straight back as `slice` — that round trip
    is the whole interaction (hover a bar, get its items; hover the next level, get
    theirs). So blanks are rendered as one stable sentinel on BOTH sides rather than
    as pandas' "nan"/"None"/"NaT" spelling of the moment, and an int band comes back
    as "1" here and "1" there instead of "1" from the dimension and "1.0" from the
    grouping.
    """
    s = series.astype(str)
    return s.mask(series.isna() | s.isin(["nan", "None", "NaT", "<NA>", ""]), UNASSIGNED)


def _dim_values(df, plan):
    kind, col, _ = plan
    if kind == "derived_year_month":
        # "2026-December" — the year normalised through Int64 so a float `year`
        # column cannot make the key "2026.0-December" on one side of the round trip
        # and "2026-December" on the other.
        y = pd.to_numeric(df["year"], errors="coerce").astype("Int64").astype(str)
        return (y + "-" + _keys(df["month"])).astype(str)
    if kind == "derived_aging":
        days = pd.to_numeric(df[col], errors="coerce")
        # Rows with no aging_days get "Unknown", never a silent sweep into 365+.
        out = pd.Series("Unknown", index=df.index, dtype=object)
        for lo, hi, lab in _AGING_BANDS:
            out = out.mask((days > lo) & (days <= hi), lab)
        return out.astype(str)
    v = _keys(df[col])
    # Integer-valued keys (priority_band) must not read as "1.0" on either side.
    if df[col].dtype.kind in "fiu":
        v = v.str.replace(r"\.0$", "", regex=True)
    return v


def _by_plan(df, by: str):
    """(key column, label column | None) for `by` on this frame, or None.

    A DERIVED dimension is groupable too — "break this category down by age band" is
    as reasonable a question as "…by material" — so the aging ladder is materialised
    as a real column rather than being refused for not existing in the parquet.
    """
    if by in ("aging_band", "aging"):
        p = _dim_plan(df, "aging_band")
        return ("__aging_band", None, p) if p else None
    if by == "year_month":
        # Groupable for the same reason the aging ladder is — and load-bearing for the
        # frontend self-check, which enumerates a chart's slices by asking for
        # `by = <the chart's own dim>`. Without this, `dim=year_month` would silently
        # degrade to "top materials", the check would drill into a MATERIAL id as if it
        # were a month, get a legitimate empty list, and report a working chart broken.
        p = _dim_plan(df, "year_month")
        return ("__year_month", None, p) if p else None
    if by in _BY_DIRECT:
        keys, labels = _BY_DIRECT[by]
    else:
        keys, labels = _DIM_DIRECT.get(by, (by,)), None
    kcol = _first_col(df, keys)
    if kcol is None:
        return None
    return kcol, (_first_col(df, labels) if labels else None), None


def _measure_plan(df, source: str, requested, primary: str = None):
    """Measure column to rank by on this frame, or None if it cannot be served.

    The PARENT table's default measure is tried before the source's own, so a
    re-routed drill still reports the measure name the chart is showing
    (`vendor_value`, not the identical `purchase_value` column it happens to be
    aliased to on the procurement frame).
    """
    if requested:
        return requested if requested in df.columns else None
    for m in (_DEFAULT_MEASURE.get(primary), _DEFAULT_MEASURE.get(source)):
        if m and m in df.columns:
            return m
    nums = [c for c in df.columns
            if df[c].dtype.kind in "fiu" and c not in ("year", "month_num", "sku_count")]
    return nums[0] if nums else None


def _category_applies(primary_table: str) -> bool:
    """Does the PARENT chart apply the global Category filter?

    The drill mirrors the chart, it never out-filters it. kpi_aging_distribution,
    kpi_vendor_volume and kpi_purchase_value carry no derived material category, so
    /kpi/<key> ignores Category on them — and a drill that DID filter would return a
    slice total smaller than the bar the user is pointing at. Judged on the PARENT
    table, so re-routing to a richer-grain source (which does carry the column) can
    never silently change the answer under the user.
    """
    # kpi_aging_distribution has no category column of its own, but /kpi/<key> now
    # serves it from the fact-grain frame whenever a Category is passed (see
    # _CATEGORY_REGRAIN), so the chart DOES narrow — and the drill has to narrow with
    # it or the panel would print a bigger total than the bar it was launched from.
    # Same for the six procurement aggregates rebuilt from fact_po / fact_grn: the bar
    # is now Rs 236.7 Cr of onco spend, so the drill inside it must be too.
    if primary_table in _CATEGORY_REGRAIN or primary_table in ds.PROC_REGRAIN:
        return True
    try:
        # load_source, not da.load: a primary table can now BE a built frame
        # (wastage-rate resolves to drill_expired_grain), and da.load would raise on
        # the name, quietly answering "no" — i.e. the drill would ignore a Category
        # the card above it is applying and print a larger total than the bar.
        # Identical to da.load for every parquet-named primary, which is all the rest.
        df = ds.load_source(primary_table, None)
    except Exception:
        return False
    return "category" in df.columns and da._is_derived_category_col(df)


def _cat_col(source: str) -> str:
    return ds.DERIVED_CATEGORY_COL.get(source, "category")


def _scoped(source: str, plant_code, category, cat_applies: bool):
    df = ds.load_source(source, plant_code)
    df = da.filter_plant(df, plant_code)
    if cat_applies:
        cat = da.resolve_category(category)
        col = _cat_col(source)
        if cat and col in df.columns:
            df = df[df[col].astype(str) == cat]
    return df


def _resolve_drill(kpi_key: str, dim, by, measure, plant_code, category):
    """Pick the first source in the chain that can actually answer the request.

    Falls back on `by` — never on the dimension or the slice, which would answer a
    different question while looking like an answer to this one — and reports it when
    it does.
    """
    sources = _sources_for(kpi_key)
    cat_applies = _category_applies(sources[0])
    wants_cat = cat_applies and da.resolve_category(category) is not None
    partial, tried = None, []
    for src in sources:
        # A source that cannot honour an ACTIVE category is skipped outright rather
        # than silently answering unfiltered. kpi_aging_distribution is exactly that
        # case: the chart narrows (via _CATEGORY_REGRAIN) but the pre-aggregated
        # parquet has no category column, so serving the drill from it would print a
        # slice total LARGER than the bar the user is pointing at.
        try:
            if wants_cat and _cat_col(src) not in ds.load_source(src, plant_code).columns:
                continue
            df = _scoped(src, plant_code, category, cat_applies)
        except Exception:
            continue
        tried.append(src)
        dplan = _dim_plan(df, dim) if dim else ("none", None, None)
        if dplan is None:
            continue
        meas = _measure_plan(df, src, measure, sources[0])
        if meas is None:
            continue
        bplan = _by_plan(df, by)
        # `by == dim` is NOT rejected here: with no slice it is how a caller
        # enumerates the chart itself ("list the material groups"), and with a slice
        # it is the degenerate one-row answer the caller explicitly asked for. The
        # self-reference is only avoided when CHOOSING a fallback grain below, where
        # the caller expressed no preference.
        if bplan is not None:
            return src, df, dplan, bplan, meas, []
        if partial is None:
            partial = (src, df, dplan, meas)

    if partial is None:
        if dim:
            raise HTTPException(
                status_code=400,
                detail=(f"Dimension '{dim}' is not available for KPI '{kpi_key}' on any of "
                        f"its sources ({', '.join(tried) or 'none'}). "
                        f"GET /drill/matrix?kpi={kpi_key} lists what it does support."))
        raise HTTPException(status_code=400,
                            detail=f"No usable measure '{measure}' for KPI '{kpi_key}'")

    # `by` could not be served anywhere — degrade to the best entity grain the chosen
    # source does have, and label the degradation in the response.
    src, df, dplan, meas = partial
    for cand in _BY_FALLBACK_ORDER:
        bplan = _by_plan(df, cand)
        if bplan is None or cand == by or (dplan[1] and bplan[0] == dplan[1]):
            continue
        note = (f"'{by}' has no grain on KPI '{kpi_key}'; showing top {cand} instead "
                f"(from {src}).")
        return src, df, dplan, bplan, meas, [("by", cand, note)]
    raise HTTPException(
        status_code=400,
        detail=(f"KPI '{kpi_key}' has no entity grain to break '{dim or 'the chart'}' down "
                f"by — its source ({src}) carries no material, vendor, category, department "
                f"or plant column, so there is nothing to drill into."))


def _name_map(kcol: str) -> dict:
    """Fallback for `name` when the frame carries no label column of its own.

    Deliberately NOT including plant: `by=plant` has never had a label column, so
    filling one in here would silently rewrite `name` from "AH01" to the hospital
    name on every plant drill that already worked. The human plant name is exposed
    on the additive `label` field below instead, where nothing depends on it yet.
    """
    if kcol in ("material", "material_id"):
        return ds.material_desc_map()
    if kcol == "cost_ctr":
        return ds.department_name_map()
    return {}


def _label_map(kcol: str) -> dict:
    """Human name for a key column, for the additive `label` field."""
    return ds.plant_name_map() if kcol == "plant" else _name_map(kcol)


@router.get("/drill/meta")
def drill_meta():
    """What the drill-down endpoint supports — so the frontend can wire hover
    handlers off a contract instead of hard-coding strings per chart.

    The per-KPI matrix (which dims each KPI really supports and what each `by`
    returns) lives on GET /drill/matrix, computed from the data rather than declared.
    """
    return {
        "kpis": sorted(set(list(REGISTRY.keys()) + list(_EXTRA_TABLES.keys())
                           + list(_KPI_ALIAS.keys()))),
        "aliases": _KPI_ALIAS,
        "dims": sorted(list(_DIM_DIRECT.keys()) + ["aging_band"]),
        "by": sorted(set(list(_BY_DIRECT.keys()) + list(_DIM_DIRECT.keys()))),
        "by_fallback_order": _BY_FALLBACK_ORDER,
        "default_measures": _DEFAULT_MEASURE,
        "non_additive_measures": sorted(_WEIGHTED_MEASURE.keys()),
        "sources": {k: _sources_for(k) for k in
                    sorted(set(list(REGISTRY.keys()) + list(_EXTRA_TABLES.keys())))},
        "default_n": 10,
        "matrix_endpoint": "/drill/matrix",
    }


@router.get("/drill/matrix")
def drill_matrix(kpi: Optional[str] = Query(None, description="one KPI key, or omit for all")):
    """THE CONTRACT the frontend wires off: every KPI -> the dimensions it supports ->
    the default measure -> what each `by` really returns.

    Computed by inspecting the resolved source chain, not hand-declared, so it cannot
    drift from what /drill/top-items will actually do. Anything a chart could ask for
    that is genuinely unanswerable is listed under `unsupported_by` with the reason,
    so nobody wires a dead hover.
    """
    keys = [kpi] if kpi else sorted(set(list(REGISTRY.keys()) + list(_EXTRA_TABLES.keys())))
    return {"kpis": [_matrix_for(_drill_kpi(k)) for k in keys], "default_n": 10}


@lru_cache(maxsize=64)
def _matrix_for(kpi_key: str) -> dict:
    sources = _sources_for(kpi_key)
    frames = []
    for s in sources:
        try:
            frames.append((s, ds.load_source(s, None)))
        except Exception:
            pass
    dims, by_opts, unsupported = {}, {}, {}
    for dim in sorted(list(_DIM_DIRECT.keys()) + ["aging_band", "year_month"]):
        for s, df in frames:
            p = _dim_plan(df, dim)
            if p:
                dims[dim] = {"source": s, "column": p[1]}
                break
    for b in sorted(set(list(_BY_DIRECT.keys()) + ["month", "year", "year_month", "sloc"])):
        hit = None
        for s, df in frames:
            p = _by_plan(df, b)
            if p:
                hit = (s, p)
                break
        if hit:
            by_opts[b] = {"source": hit[0], "key": hit[1][0], "label": hit[1][1]}
        else:
            unsupported[b] = ("no " + b + " grain on "
                              + " / ".join(s for s, _ in frames))
    primary = sources[0]
    meas = _DEFAULT_MEASURE.get(primary)
    if not meas and frames:
        meas = _measure_plan(frames[0][1], primary, None, primary)

    # A KPI with no ENTITY grain anywhere in its chain cannot be drilled at all —
    # forecast_accuracy is six (metric, value) rows, so there is nothing inside a bar
    # to list. It still has `dims` in the mechanical sense (a `metric` column exists to
    # group by), and advertising those was the contract lying: the frontend would wire
    # a hover off a dimension the endpoint then 400s on, which is the exact "a broken
    # chart looks identical to an unwired one" failure this matrix exists to prevent.
    # So a non-drillable KPI advertises NOTHING, and says why.
    entity_by = [b for b in by_opts if b in _BY_DIRECT]
    drillable = bool(entity_by)
    if not drillable:
        dims = {}
    return {
        "kpi": kpi_key, "primary_table": primary, "sources": sources,
        "drillable": drillable,
        "not_drillable_reason": (None if drillable else
                                 f"{primary} carries no material, vendor, plant, category, "
                                 f"department or batch column — there is no entity inside "
                                 f"a slice to list."),
        "default_measure": meas,
        "measure_additive": meas not in _WEIGHTED_MEASURE,
        "respects_category": _category_applies(primary),
        "dims": dims, "by": by_opts, "unsupported_by": unsupported,
    }


@router.get("/drill/top-items")
def drill_top_items(
    kpi: str = Query(..., description="registry KPI key (see /meta/kpis) or a /drill/meta extra"),
    dim: Optional[str] = Query(None, description="dimension the chart is grouped by"),
    slice_value: Optional[str] = Query(None, alias="slice",
                                       description="the specific slice value hovered; omit = whole chart"),
    by: str = Query("material", description="what the returned items are"),
    measure: Optional[str] = Query(None, description="numeric column to rank by"),
    n: int = Query(10, ge=1, le=200),
    plant: Optional[str] = Query(None, alias="Plant"),
    category: Optional[str] = Query(None, alias="Category"),
):
    """Top-N items inside one chart slice.

    The answer to "ok, this bar is Rs 25 Cr of Onco Drugs — WHAT is in it?".
    `share_pct` is the item's share OF THE HOVERED SLICE (not of the portfolio) and
    `cum_share_pct` lets a tooltip say "these 10 are 62% of this bar". `covered_pct`
    is the same figure for the whole returned set, so the reader always knows how
    much of the slice the list actually accounts for rather than assuming it is all
    of it.

    When the KPI's own table cannot answer at this grain the request is re-routed to
    the fact frame that table was BUILT from (`source` says which, `rerouted` flags
    it). When the requested `by` exists on no source the response degrades to the
    best available grain and reports `by_actual` / `by_fallback` / `note` — a
    drill-down that silently returns nothing is worse than one that says what it
    could do instead.
    """
    kpi_key = _drill_kpi(kpi)
    # A drill MIRRORS its chart. For the one KPI whose chart refuses a category cut, an
    # unfiltered top-10 under an "Onco Drugs" label would be the same silent lie the 400
    # on /kpi/<key> exists to prevent — worse here, because there is no bar above it to
    # compare the total against.
    if kpi_key in REGISTRY:
        _category_unsupported(REGISTRY[kpi_key][0], category)
    plant_code = da.resolve_plant(plant)
    source, df, dplan, bplan, meas, notes = _resolve_drill(
        kpi_key, dim, by, measure, plant_code, category)
    kcol, lcol, by_derived = bplan

    weights = _WEIGHTED_MEASURE.get(meas)
    wcol = _first_col(df, weights) if weights is not None else None
    additive = weights is None

    df = df.copy()
    if by_derived is not None:                 # e.g. by=aging_band — materialise it
        df[kcol] = _dim_values(df, by_derived)
    df[meas] = pd.to_numeric(df[meas], errors="coerce").fillna(0.0)
    if not additive:
        df["_w"] = (pd.to_numeric(df[wcol], errors="coerce").fillna(0.0) if wcol
                    else pd.Series(1.0, index=df.index))

    def _agg(frame):
        if additive:
            return float(frame[meas].sum())
        if not len(frame):
            return 0.0
        w = float(frame["_w"].sum())
        return float((frame[meas] * frame["_w"]).sum() / w) if w > 0 else float(frame[meas].mean())

    grand_total = _agg(df)

    resolved_dim = dplan[2]
    if dplan[0] != "none" and slice_value is not None:
        df = df[_dim_values(df, dplan) == str(slice_value)]

    # dropna=False is load-bearing, not tidiness. Grouping by [key, label] with
    # pandas' default dropna SILENTLY DELETED every row whose label was null — 9,889
    # rows of kpi_monthly_purchase_value and 8,839 of kpi_stock_change, materials
    # missing from dim_material. The drill then reported Rs 433 Cr as "100% of this
    # bar" when the bar was Rs 650 Cr: a third of the money vanished, and the
    # `covered_pct` disclosure that exists precisely to stop that kind of overclaim
    # was itself computed off the truncated total.
    gcols = [kcol] + ([lcol] if lcol and lcol != kcol else [])
    if additive:
        g = df.groupby(gcols, as_index=False, observed=True, dropna=False)[meas].sum()
    else:
        df["_wm"] = df[meas] * df["_w"]
        gg = df.groupby(gcols, as_index=False, observed=True, dropna=False)[
            ["_wm", "_w", meas]].agg({"_wm": "sum", "_w": "sum", meas: "mean"})
        gg[meas] = np.where(gg["_w"] > 0, gg["_wm"] / gg["_w"].replace(0, np.nan), gg[meas])
        g = gg[gcols + [meas]]

    # Totalled over the GROUPED frame, matching the summation order the shares are
    # computed against, so share_pct over the whole slice sums to exactly 100.
    slice_total = float(g[meas].sum()) if additive else _agg(df)
    g = g.sort_values(meas, ascending=False).head(int(n))

    # Read column-wise, never through iterrows(): a row Series takes ONE dtype for
    # the whole row, so an int key next to a float measure came back as "1.0" — a key
    # that then matched nothing when the caller passed it straight back as `slice`.
    # Rows the source has no key for (service POs with no material code, batches with
    # no batch number) are NAMED for what they are rather than shown as "nan" or
    # dropped: on monthly-purchase-value that group is Rs 173 Cr, the single largest
    # line in the chart.
    name_map, lab_map = _name_map(kcol), _label_map(kcol)
    keys = _dim_values(g, ("direct", kcol, kcol)).tolist()
    labs = (_keys(g[lcol]).tolist() if lcol and lcol != kcol else None)
    vals = [float(v) for v in g[meas].tolist()]

    items, cum = [], 0.0
    for i, (key, v) in enumerate(zip(keys, vals), start=1):
        cum += v
        lab = labs[i - 1] if labs else None
        name = lab if (lab and lab != UNASSIGNED) else (name_map.get(key) or key)
        items.append({
            "rank": i, "key": key, "name": name, "label": lab_map.get(key, name),
            "value": v,
            "share_pct": (round(v / slice_total * 100, 2) if slice_total else 0.0) if additive else None,
            "cum_share_pct": (round(cum / slice_total * 100, 2) if slice_total else 0.0) if additive else None,
        })

    note_by = next((c for k, c, _ in notes if k == "by"), None)
    note_txt = "; ".join(t for _, _, t in notes) or None
    if not additive:
        approx = " (approximate)" if meas in _APPROXIMATE_MEASURE else ""
        agg_note = (f"'{meas}' is not additive — items are ranked by a "
                    f"{'weighted mean' if wcol else 'mean'}{approx}"
                    + (f" weighted by {wcol}" if wcol else "")
                    + ". Shares are omitted: a percentage of an average is meaningless.")
        note_txt = f"{note_txt} {agg_note}" if note_txt else agg_note

    return {
        "kpi": kpi, "kpi_resolved": kpi_key,
        "table": source, "source": source, "primary_table": _sources_for(kpi_key)[0],
        "rerouted": source != _sources_for(kpi_key)[0],
        "dim": resolved_dim, "dim_requested": dim, "slice": slice_value,
        "by": note_by or by, "by_requested": by, "by_actual": note_by or by,
        "by_fallback": bool(note_by),
        "measure": meas, "measure_requested": measure,
        "additive": additive,
        "agg": "sum" if additive else ("weighted_mean" if wcol else "mean"),
        "weight": wcol, "n": int(n),
        "slice_total": slice_total, "grand_total": grand_total,
        "slice_share_pct": ((round(slice_total / grand_total * 100, 2) if grand_total else 0.0)
                            if additive else None),
        "count": int(len(df[kcol].astype(str).unique())), "returned": len(items),
        "covered_pct": ((round(cum / slice_total * 100, 2) if slice_total else 0.0)
                        if additive else None),
        "note": note_txt,
        "items": items,
    }
