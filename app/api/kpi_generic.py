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


@router.get("/kpi/{key}")
def kpi_chart(
    key: str,
    plant: Optional[str] = Query(None, alias="Plant"),
    material: Optional[str] = Query(None, alias="Material"),
    material_group: Optional[str] = Query(None, alias="MaterialGroup"),
    group_by: Optional[str] = Query(None, description="comma cols to aggregate by"),
    measures: Optional[str] = Query(None, description="comma numeric cols to sum"),
    top: Optional[int] = Query(None, description="keep top-N rows by first measure"),
):
    table = _resolve(key)
    return da.chart_series(table, plant=plant, material=material, material_group=material_group,
                           group_by=group_by, measures=measures, top=top)


@router.get("/kpi/{key}/summary")
def kpi_summary(
    key: str,
    plant: Optional[str] = Query(None, alias="Plant"),
    material: Optional[str] = Query(None, alias="Material"),
    material_group: Optional[str] = Query(None, alias="MaterialGroup"),
):
    table = _resolve(key)
    return da.summarize(table, plant=plant, material=material, material_group=material_group)


@router.get("/kpi/{key}/table")
def kpi_table(
    key: str,
    request: Request,
    plant: Optional[str] = Query(None, alias="Plant"),
):
    table = _resolve(key)
    params = dict(request.query_params)
    return da.paginate(table, plant, params, col_map={})


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
def portfolio_summary(name: str, plant: Optional[str] = Query(None, alias="Plant")):
    keys = PORTFOLIOS.get(name)
    if not keys:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio '{name}'")
    out = {}
    for k in keys:
        table = REGISTRY[k][0]
        out[k] = da.summarize(table, plant=da.resolve_plant(plant))
    return out


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
