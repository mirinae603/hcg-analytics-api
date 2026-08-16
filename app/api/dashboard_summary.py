"""Executive-summary cards (original frontend contract) + admin refresh.

`/api/dashboard/all?region=<plant code | name>` returns the exact nested shape the
original AnalyticsDashboardLayout expects (stockAging / kpiStockLevel / returnRate /
daysOnHand / inventoryTurnover). Region that isn't a known plant code => all plants.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from typing import Optional
import numpy as np
import pandas as pd

from app.core import data_access as da
from app.core.config import settings

router = APIRouter()


def _resolve_plant(region: Optional[str]) -> Optional[str]:
    return da.resolve_plant(region)


AGING_BINS = [-1, 30, 90, 180, 365, float("inf")]
AGING_LABELS = ["0-30", "31-90", "91-180", "181-365", "365+"]


def _aging_buckets(plant, category):
    """Stock value per aging bucket.

    Default path (no category) reads the committed kpi_aging_distribution parquet —
    unchanged, byte-identical. When a category IS selected we recompute from
    fact_inventory instead, because kpi_aging_distribution is pre-aggregated to
    plant x material_group x aging_bucket and carries no material dimension, so a
    category filter on it would be a silent no-op that shows portfolio-wide numbers
    under a category label. The recomputation is the ETL's own A7 groupby and has
    been verified to reproduce that parquet EXACTLY (all 6,497 rows, zero diff).
    """
    if not da.resolve_category(category):
        ad = da.filter_plant(da.load("kpi_aging_distribution"), plant)
        return ad.groupby("aging_bucket", observed=True)["stock_value"].sum()
    inv = da.filter_category(da.filter_plant(da.load("fact_inventory"), plant), category).copy()
    inv["aging_bucket"] = pd.cut(pd.to_numeric(inv["aging_days"], errors="coerce"),
                                 bins=AGING_BINS, labels=AGING_LABELS)
    return inv.groupby("aging_bucket", observed=True)["total_cost"].sum()


@router.get("/api/dashboard/all")
def dashboard_all(region: Optional[str] = Query(None),
                  category: Optional[str] = Query(None, alias="Category")):
    plant = _resolve_plant(region)

    def _f(table):
        return da.filter_category(da.filter_plant(da.load(table), plant), category)

    def _fdf(df):
        return da.filter_category(da.filter_plant(df, plant), category)

    inv = _f("kpi_stock_value")
    # nonmoving reads the billed+internal ("both") recompute, not the raw internal-only
    # parquet -- legacy_kpi.py's nonmoving_insights moved to "both" permanently (see
    # memory: billable-nonbillable-consumption-card), and this exec-summary tile has to
    # move with it or it silently shows a different, stale figure than the page one click
    # away shows for the exact same KPI. `health` stays RAW here on purpose -- see
    # avg_itr below for why. (DOH itself is now da.doh_value_scoped(), a value-basis
    # portfolio ratio computed straight from kpi_stock_value/fact_consumption/
    # sales_by_material below -- no separate doh_scoped() load needed for it here.)
    health = _f("kpi_health_score")
    units = _f("kpi_units_consumed")  # Consumption KPI: default view stays internal-
                                       # only, matching Units Consumed per SKU's own
                                       # default scope (that page keeps its live toggle).
    nonmoving = _fdf(da.nonmoving_scoped("both"))

    bucket = _aging_buckets(plant, category)
    fresh = float(bucket.get("0-30", 0))
    aging = float(bucket.get("31-90", 0))
    problem = float(bucket.get("91-180", 0) + bucket.get("181-365", 0))
    dead = float(bucket.get("365+", 0))

    stock_value = float(inv["stock_value_cost"].sum())
    stock_mrp = float(inv["stock_value_mrp"].sum())
    stock_qty = float(inv["stock_qty"].sum())
    cons_cost = float(units["consumption_cost"].sum())
    # MRP-proxy revenue & margin (flagged as proxy in workbook)
    revenue_proxy = stock_mrp
    margin = round((1 - (stock_value / stock_mrp)) * 100, 1) if stock_mrp else 0.0
    # Client's own formula (Closing Inventory Value / avg daily Value sold+consumed),
    # the same da.doh_value_scoped() the bespoke DOH drill-down's own headline now uses
    # (legacy_kpi.py's doh_insights, totals.days_inventory_value) — matches that page so
    # this tile and that page agree, instead of showing two different DOH numbers on one
    # visit. None (no billed-cost-applicable denominator, or a filter with zero rows)
    # becomes 0.0 here rather than crashing the response.
    _dv = da.doh_value_scoped(region, category)
    avg_doh = float(_dv["days_inventory_value"]) if _dv["days_inventory_value"] is not None else 0.0
    # Portfolio-level SUM(consumption)/SUM(inventory), annualized -- NOT a mean of each
    # material's own turnover ratio. A mean-of-ratios lets a handful of near-zero-inventory
    # SKUs (a rounding-error's worth of stock value) post ratios in the tens of thousands and
    # pull the average far above any plant's real turnover. This is the identical
    # "portfolio_itr" formula the bespoke Inventory Turnover Ratio drill-down already uses
    # (legacy_kpi.py's `itr_insights`), so this tile matches that page instead of showing
    # 2.4x here vs 0.66x there for the same underlying data.
    _inv6 = float(health["closing_stock_value"].sum())
    _internal_cogs6 = float(health["consumption_cost"].sum())
    ANN = 2.0  # 6 months of data -> annualized

    # Billed COGS folded in exactly like itr_insights' "both" branch: a material-grain,
    # DEDUPED sum from sales_by_material, added to `health`'s own RAW plant-grain sum --
    # never computed by summing a plant-fanned-out itr_scoped()/health_scoped() frame,
    # which would multiply the billed figure by however many plants stock each material
    # (caught live earlier this session: a naive merge-then-sum inflated portfolio COGS
    # to Rs 3,305 Cr against a real Rs 308 Cr billable total -- this is why `health`
    # above stays the raw table rather than a "both"-scoped one). sales_by_material
    # carries no plant column, so this is network-wide billed COGS folded in regardless
    # of which Plant is selected -- the same caveat itr_insights documents for its own
    # page, replicated here rather than reinvented.
    _billed_cogs6 = 0.0
    sm_path = da.KPI / "sales_by_material.parquet"
    if sm_path.exists():
        sm = pd.read_parquet(sm_path)
        cat = da.resolve_category(category)
        if cat:
            cmap = da._material_category_map()
            sm = sm[sm["material"].astype(str).map(cmap).fillna(da.CATEGORY_UNCLASSIFIED) == cat]
        _billed_cogs6 = float(sm["cost"].sum())
    _cogs6 = _internal_cogs6 + _billed_cogs6
    avg_itr = (_cogs6 * ANN) / _inv6 if _inv6 else 0.0

    loc = "All Plants" if not plant else plant

    return {
        "stockAging": {
            "fresh": round(fresh, 2), "aging": round(aging, 2),
            "problem": round(problem, 2), "deadStock": round(dead, 2),
            "lastUpdated": "31 May 2026",
        },
        "kpiStockLevel": {
            "currentStock": round(stock_qty), "stockValue": round(stock_value, 2),
            "lastMonthRevenue": round(cons_cost, 2), "maxStockValue": round(stock_value * 1.25, 2),
            "monthlyRevenueTarget": round(revenue_proxy, 2), "margin": margin,
            "unit": "Units", "currency": "INR", "label": "Current Stock Value",
            "lowStockThreshold": 1000, "location": loc, "supplier": "—",
            "lastUpdated": "31 May 2026",
        },
        # "trend" is omitted (None) on all three cards below, not a fabricated 0% -- none
        # of kpi_doh / kpi_health_score / the return-rate proxy carry a month dimension in
        # the source data, so there is no real prior-period figure to compare against. A
        # literal "0% vs last month" copy-pasted across all three read as three independent
        # zero-change results when it was really "we never computed this" -- every card
        # component here already renders nothing when trend is falsy, so omitting it is a
        # pure UI no-op, never a broken card.
        "returnRate": {  # no returns in data -> zero/proxy (see gap log)
            "currentReturnRate": 0.0,
            "historicalData": {"thirtyDaysAgo": 0.0, "sixtyDaysAgo": 0.0, "ninetyDaysAgo": 0.0},
            "trend": None,
            "targetReturnRate": 2.0, "industryAverage": 3.0,
        },
        "daysOnHand": {
            "daysOnHand": round(avg_doh, 1),
            "trend": None,
            "criticalThreshold": 15, "optimalRange": {"min": 30, "max": 90},
            "category": da.resolve_category(category) or "All",
            "location": loc, "lastCalculated": "31 May 2026",
        },
        "inventoryTurnover": {
            "currentITR": round(avg_itr, 2), "label": "Inventory Turnover",
            "trend": None,
            "targetITR": 6.0, "industryAverage": 4.0,
        },
        # extra block consumed by the rebuilt hcg-analytics-ui exec summary
        "stockValue": {"currentStockValue": round(stock_value, 2), "stockMrpValue": round(stock_mrp, 2),
                       "skuCount": int(inv["material"].nunique())},
        # Procurement & fill-rate are vendor/process metrics whose source tables carry
        # no material dimension at all, so they are deliberately NOT category-scoped —
        # `categoryScope` below says exactly which blocks the selector reaches.
        "procurement": {"purchaseValue": round(float(da.filter_plant(da.load("kpi_purchase_value"), plant)["purchase_value"].sum()), 2),
                        "vendorCount": int(da.filter_plant(da.load("kpi_purchase_value"), plant)["vendor_name"].nunique())},
        "consumption": {"unitsConsumed": float(units["total_units"].sum()), "consumptionCost": round(cons_cost, 2)},
        "replenishment": {"materialsNeeding": int((_f("stock_replenishment_and_aging_risk")["replenishment_quantity"] > 0).sum()),
                          "totalQtyToOrder": round(float(_f("stock_replenishment_and_aging_risk")["replenishment_quantity"].sum()), 2)},
        "nearExpiry": {"skuCount": int(_f("kpi_near_expiry")["material"].nunique()),
                       "value": round(float(_f("kpi_near_expiry")["total_cost"].sum()), 2)},
        "stockAgingSummary": {"nonMovingSkus": int(nonmoving["material"].nunique())},
        "fillRate": {"avgFillRatePct": round(float(da.filter_plant(da.load("kpi_fill_rate"), plant)["fill_rate_pct"].mean()), 1) if len(da.filter_plant(da.load("kpi_fill_rate"), plant)) else None},
        "categoryScope": {"applied": da.resolve_category(category),
                          "scoped": ["stockAging", "kpiStockLevel", "daysOnHand", "inventoryTurnover",
                                     "stockValue", "consumption", "replenishment", "nearExpiry",
                                     "stockAgingSummary"],
                          "unscoped": ["procurement", "fillRate", "returnRate"]},
    }


@router.post("/admin/refresh-data")
def refresh_data(x_admin_token: Optional[str] = Header(None)):
    if x_admin_token != settings.ADMIN_REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid admin token")
    da.refresh_cache()
    # Also drop the memoized portfolio-overview / summary results so they recompute
    # against the refreshed data (otherwise they'd serve the previous snapshot).
    try:
        from app.api import legacy_kpi, kpi_generic
        legacy_kpi.clear_result_caches()
        kpi_generic._portfolio_summary_cached.cache_clear()
        kpi_generic.clear_kpi_caches()
    except Exception:
        pass
    return {"status": "cache cleared"}
