"""Executive-summary cards (original frontend contract) + admin refresh.

`/api/dashboard/all?region=<plant code | name>` returns the exact nested shape the
original AnalyticsDashboardLayout expects (stockAging / kpiStockLevel / returnRate /
daysOnHand / inventoryTurnover). Region that isn't a known plant code => all plants.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from typing import Optional
import numpy as np

from app.core import data_access as da
from app.core.config import settings

router = APIRouter()


def _resolve_plant(region: Optional[str]) -> Optional[str]:
    return da.resolve_plant(region)


@router.get("/api/dashboard/all")
def dashboard_all(region: Optional[str] = Query(None)):
    plant = _resolve_plant(region)

    inv = da.filter_plant(da.load("kpi_stock_value"), plant)
    agedist = da.filter_plant(da.load("kpi_aging_distribution"), plant)
    doh = da.filter_plant(da.load("kpi_doh"), plant)
    health = da.filter_plant(da.load("kpi_health_score"), plant)
    units = da.filter_plant(da.load("kpi_units_consumed"), plant)
    nonmoving = da.filter_plant(da.load("kpi_non_moving"), plant)

    bucket = agedist.groupby("aging_bucket", observed=True)["stock_value"].sum()
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
    avg_doh = float(doh["doh_days"].replace([np.inf, -np.inf], np.nan).dropna().mean() or 0)
    avg_itr = float(health["turnover_annualized"].replace([np.inf, -np.inf], np.nan).dropna().mean() or 0)

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
        "returnRate": {  # no returns in data -> zero/proxy (see gap log)
            "currentReturnRate": 0.0,
            "historicalData": {"thirtyDaysAgo": 0.0, "sixtyDaysAgo": 0.0, "ninetyDaysAgo": 0.0},
            "trend": {"direction": "stable", "percentage": 0, "period": "vs last month"},
            "targetReturnRate": 2.0, "industryAverage": 3.0,
        },
        "daysOnHand": {
            "daysOnHand": round(avg_doh, 1),
            "trend": {"direction": "stable", "percentage": 0, "period": "vs last month"},
            "criticalThreshold": 15, "optimalRange": {"min": 30, "max": 90},
            "category": "All", "location": loc, "lastCalculated": "31 May 2026",
        },
        "inventoryTurnover": {
            "currentITR": round(avg_itr, 2), "label": "Inventory Turnover",
            "trend": {"direction": "stable", "percentage": 0, "period": "vs last month"},
            "targetITR": 6.0, "industryAverage": 4.0,
        },
        # extra block consumed by the rebuilt hcg-analytics-ui exec summary
        "stockValue": {"currentStockValue": round(stock_value, 2), "stockMrpValue": round(stock_mrp, 2),
                       "skuCount": int(inv["material"].nunique())},
        "procurement": {"purchaseValue": round(float(da.filter_plant(da.load("kpi_purchase_value"), plant)["purchase_value"].sum()), 2),
                        "vendorCount": int(da.filter_plant(da.load("kpi_purchase_value"), plant)["vendor_name"].nunique())},
        "consumption": {"unitsConsumed": float(units["total_units"].sum()), "consumptionCost": round(cons_cost, 2)},
        "replenishment": {"materialsNeeding": int((da.filter_plant(da.load("stock_replenishment_and_aging_risk"), plant)["replenishment_quantity"] > 0).sum()),
                          "totalQtyToOrder": round(float(da.filter_plant(da.load("stock_replenishment_and_aging_risk"), plant)["replenishment_quantity"].sum()), 2)},
        "nearExpiry": {"skuCount": int(da.filter_plant(da.load("kpi_near_expiry"), plant)["material"].nunique()),
                       "value": round(float(da.filter_plant(da.load("kpi_near_expiry"), plant)["total_cost"].sum()), 2)},
        "stockAgingSummary": {"nonMovingSkus": int(nonmoving["material"].nunique())},
        "fillRate": {"avgFillRatePct": round(float(da.filter_plant(da.load("kpi_fill_rate"), plant)["fill_rate_pct"].mean()), 1) if len(da.filter_plant(da.load("kpi_fill_rate"), plant)) else None},
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
