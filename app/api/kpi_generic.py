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

import pandas as pd

from app.core import data_access as da

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


@_kc
def _kpi_chart_cached(table, plant, material, material_group, group_by, measures, top, category=None):
    return da.chart_series(table, plant=plant, material=material, material_group=material_group,
                           group_by=group_by, measures=measures, top=top, category=category)


@_kc
def _kpi_summary_cached(table, plant, material, material_group, category=None):
    return da.summarize(table, plant=plant, material=material, material_group=material_group,
                        category=category)


# `Category` is OPTIONAL everywhere and defaults to None => filter_category is a
# no-op, so every existing call site keeps its exact current response. On tables
# with no material grain (kpi_purchase_value, kpi_cycle_time, kpi_fill_rate,
# kpi_aging_distribution, …) no `category` column is derived at all, so the param
# is inert there too — see data_access._attach_category.
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
    return _kpi_chart_cached(_resolve(key), plant, material, material_group, group_by,
                             measures, top, category)


@router.get("/kpi/{key}/summary")
def kpi_summary(
    key: str,
    plant: Optional[str] = Query(None, alias="Plant"),
    material: Optional[str] = Query(None, alias="Material"),
    material_group: Optional[str] = Query(None, alias="MaterialGroup"),
    category: Optional[str] = Query(None, alias="Category"),
):
    return _kpi_summary_cached(_resolve(key), plant, material, material_group, category)


@router.get("/kpi/{key}/table")
def kpi_table(
    key: str,
    request: Request,
    plant: Optional[str] = Query(None, alias="Plant"),
    category: Optional[str] = Query(None, alias="Category"),
):
    table = _resolve(key)
    params = dict(request.query_params)
    return da.paginate(table, plant, params, col_map={}, category=category)


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
    return {k: da.summarize(REGISTRY[k][0], plant=plant, category=category) for k in PORTFOLIOS[name]}


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

# dim alias -> (column, kind). "derived" dims are computed from a numeric column.
_DIM_DIRECT = {
    "material_group": "material_group",
    "plant": "plant",
    "category": "category",
    "vendor": "vendor_name",
    "month": "month",
    "year": "year",
    "department": "cost_ctr",
    "expiry_bucket": "expiry_bucket",
    "aging_category": "aging_category",
    "aging_bucket": "aging_bucket",
    "aging_risk": "aging_risk",
    "risk_level": "risk_level",
    "health_tier": "health_tier",
    "radar_status": "radar_status",
    "reason": "reason",
    "aging_risk_forecast": "aging_risk_forecast",
}

# Derived aging band — the ladder every aging chart in the UI uses.
_AGING_BANDS = [(-1, 30, "0-30"), (30, 90, "31-90"), (90, 180, "91-180"),
                (180, 365, "181-365"), (365, float("inf"), "365+")]

# by-alias -> (key column, label column | None)
_BY_DIRECT = {
    "material": (("material", "material_id"), ("material_desc", "desc")),
    "vendor": (("vendor_name",), None),
    "plant": (("plant",), None),
    "material_group": (("material_group",), None),
    "category": (("category",), None),
    "department": (("cost_ctr",), ("department_name",)),
}

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
    "stock_replenishment_and_aging_risk": "replenishment_quantity",
}

# Extra tables the drill-down can reach that are not registry KPI keys.
_EXTRA_TABLES = {"reorder-priority": "stock_replenishment_and_aging_risk",
                 "replenishment": "stock_replenishment_and_aging_risk",
                 "inventory": "fact_inventory"}


def _drill_table(kpi: str) -> str:
    if kpi in REGISTRY:
        return REGISTRY[kpi][0]
    if kpi in _EXTRA_TABLES:
        return _EXTRA_TABLES[kpi]
    raise HTTPException(status_code=404, detail=f"Unknown KPI/table '{kpi}' for drill-down")


def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _apply_dim(df, dim: str):
    """Return (series_of_dim_values, resolved_dim_name) or (None, None)."""
    if dim in ("aging_band", "aging") and "aging_days" in df.columns:
        days = pd.to_numeric(df["aging_days"], errors="coerce")
        # Rows with no aging_days get "Unknown", never a silent sweep into 365+.
        out = pd.Series("Unknown", index=df.index, dtype=object)
        for lo, hi, lab in _AGING_BANDS:
            out = out.mask((days > lo) & (days <= hi), lab)
        return out.astype(str), "aging_band"
    col = _DIM_DIRECT.get(dim, dim)
    if col in df.columns:
        return df[col].astype(str), col
    return None, None


@router.get("/drill/meta")
def drill_meta():
    """What the drill-down endpoint supports — so the frontend can wire hover
    handlers off a contract instead of hard-coding strings per chart."""
    return {
        "kpis": sorted(list(REGISTRY.keys()) + list(_EXTRA_TABLES.keys())),
        "dims": sorted(list(_DIM_DIRECT.keys()) + ["aging_band"]),
        "by": sorted(_BY_DIRECT.keys()),
        "default_measures": _DEFAULT_MEASURE,
        "default_n": 10,
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
    """
    table = _drill_table(kpi)
    df = da.filter_plant(da.load(table), da.resolve_plant(plant))
    df = da.filter_category(df, category)

    meas = measure or _DEFAULT_MEASURE.get(table)
    if not meas or meas not in df.columns:
        nums = [c for c in df.columns
                if df[c].dtype.kind in "fiu" and c not in ("year", "month_num")]
        if not nums:
            raise HTTPException(status_code=400, detail=f"No numeric measure on '{table}'")
        meas = nums[0]

    grand_total = float(pd.to_numeric(df[meas], errors="coerce").fillna(0).sum())

    resolved_dim = None
    if dim:
        vals, resolved_dim = _apply_dim(df, dim)
        if vals is None:
            raise HTTPException(status_code=400,
                                detail=f"Dimension '{dim}' not available on '{table}'")
        if slice_value is not None:
            df = df[vals == str(slice_value)]

    keys, labels = _BY_DIRECT.get(by, ((by,), None))
    kcol = _first_col(df, keys)
    if kcol is None:
        raise HTTPException(status_code=400, detail=f"Cannot group by '{by}' on '{table}'")
    lcol = _first_col(df, labels) if labels else None

    df = df.copy()
    df[meas] = pd.to_numeric(df[meas], errors="coerce").fillna(0.0)
    gcols = [kcol] + ([lcol] if lcol and lcol != kcol else [])
    g = df.groupby(gcols, as_index=False, observed=True)[meas].sum()

    slice_total = float(g[meas].sum())
    g = g.sort_values(meas, ascending=False).head(int(n))

    items, cum = [], 0.0
    for i, (_, r) in enumerate(g.iterrows(), start=1):
        v = float(r[meas])
        cum += v
        items.append({
            "rank": i, "key": str(r[kcol]),
            "name": str(r[lcol]) if lcol and lcol != kcol else str(r[kcol]),
            "value": v,
            "share_pct": round(v / slice_total * 100, 2) if slice_total else 0.0,
            "cum_share_pct": round(cum / slice_total * 100, 2) if slice_total else 0.0,
        })

    return {
        "kpi": kpi, "table": table, "dim": resolved_dim, "slice": slice_value,
        "by": by, "measure": meas, "n": int(n),
        "slice_total": slice_total, "grand_total": grand_total,
        "slice_share_pct": round(slice_total / grand_total * 100, 2) if grand_total else 0.0,
        "count": int(len(df[kcol].astype(str).unique())), "returned": len(items),
        "covered_pct": round(cum / slice_total * 100, 2) if slice_total else 0.0,
        "items": items,
    }
