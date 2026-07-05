"""Replenishment & aging-risk dashboard (D1 Replenishment Required, D4 Qty to Order,
D8 SKU Replenishment Monitoring).

Reads the ETL-produced `stock_replenishment_and_aging_risk` parquet. Response model is
unchanged from the original so the existing frontend page renders without changes.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import logging
import pandas as pd

from app.core import data_access as da

router = APIRouter()


class TopReplenishmentItem(BaseModel):
    material_id: str
    replenishment_quantity: float
    aging_risk: str
    closing_stock: float
    demand_forecast: float


class TopAgingRiskItem(BaseModel):
    material_id: str
    aging_risk: str
    closing_stock: float
    demand_forecast: float
    replenishment_quantity: float


class SummaryStats(BaseModel):
    total_materials: int
    total_closing_stock: float
    total_replenishment_needed: float
    materials_needing_replenishment: int
    materials_with_aging_risk: int
    avg_demand_forecast: float
    aging_risk_breakdown: Dict[str, int]


class DashboardResponse(BaseModel):
    top_replenishment: List[TopReplenishmentItem]
    top_aging_risk: List[TopAgingRiskItem]
    summary_stats: SummaryStats
    total_records: int
    filters_applied: Dict[str, Any]


@router.get("/inventory/replenishment-data", response_model=DashboardResponse)
def get_replenishment_dashboard_data(
    plant: Optional[str] = Query(None),
    material_id: Optional[str] = Query(None),
):
    try:
        df = da.load("stock_replenishment_and_aging_risk").copy()
        df = da.filter_plant(df, plant)
        filters_applied: Dict[str, Any] = {}
        if plant:
            filters_applied["plant"] = plant
        if material_id:
            df = df[df["material_id"].astype(str).str.contains(material_id, case=False, na=False)]
            filters_applied["material_id"] = material_id

        df = df.fillna(0)
        name_of = df.set_index("material_id")["material_desc"].to_dict() if "material_desc" in df.columns else {}

        summary = SummaryStats(
            total_materials=len(df),
            total_closing_stock=float(df["closing_stock"].sum()),
            total_replenishment_needed=float(df["replenishment_quantity"].sum()),
            materials_needing_replenishment=int((df["replenishment_quantity"] > 0).sum()),
            materials_with_aging_risk=int((df["inventory_aging_risk"] == True).sum()),
            avg_demand_forecast=float(df["demand_forecast"].mean()) if len(df) else 0.0,
            aging_risk_breakdown={str(k): int(v) for k, v in df["aging_risk"].value_counts().to_dict().items()},
        )

        top_rep = df[df["replenishment_quantity"] > 0].nlargest(20, "replenishment_quantity")
        top_age = df[df["aging_risk"].astype(str).str.contains("1\\+", na=False)].nlargest(20, "closing_stock")

        def label(r):
            return str(name_of.get(r["material_id"], r["material_id"]))

        return DashboardResponse(
            top_replenishment=[TopReplenishmentItem(
                material_id=label(r), replenishment_quantity=float(r["replenishment_quantity"]),
                aging_risk=str(r["aging_risk"]), closing_stock=float(r["closing_stock"]),
                demand_forecast=float(r["demand_forecast"])) for _, r in top_rep.iterrows()],
            top_aging_risk=[TopAgingRiskItem(
                material_id=label(r), aging_risk=str(r["aging_risk"]),
                closing_stock=float(r["closing_stock"]), demand_forecast=float(r["demand_forecast"]),
                replenishment_quantity=float(r["replenishment_quantity"])) for _, r in top_age.iterrows()],
            summary_stats=summary, total_records=len(df), filters_applied=filters_applied,
        )
    except Exception as e:
        logging.exception("replenishment error")
        raise HTTPException(status_code=500, detail=str(e))


# ── "Generate Forecast Report" panel (ForecastInsights) ───────────────────────
# The replenishment page's timeframe selector requests a per-material demand &
# cash-flow projection over a 7/14/21/28-day horizon. Weekly demand = monthly
# forecast scaled by (horizon days / 30); cash-flow = demand × unit cost.

WEEK_DAYS = {"week_1": 7, "week_2": 14, "week_3": 21, "week_4": 28}


class FilteredDataResponse(BaseModel):
    data: List[Dict[str, Any]]
    week_filter: str
    plant_filter: Optional[str] = None
    total_records: int


@router.get("/api/weeks")
def get_available_weeks():
    return {"weeks": ["week_1", "week_2", "week_3", "week_4"]}


@router.get("/api/data", response_model=FilteredDataResponse)
def get_weekly_forecast_data(
    week: str = Query("week_1"),
    plant: Optional[str] = Query(None),
    limit: int = Query(250, ge=1, le=2000),
):
    try:
        df = da.load("stock_replenishment_and_aging_risk").copy()
        df = da.filter_plant(df, plant)
        df = df.fillna(0)
        if not len(df):
            return FilteredDataResponse(data=[], week_filter=week, plant_filter=plant, total_records=0)

        # collapse plant rows → one row per material (covers the "All Plants" case)
        agg = df.groupby("material_id", as_index=False).agg(
            material_desc=("material_desc", "first"),
            closing_stock=("closing_stock", "sum"),
            closing_stock_value=("closing_stock_value", "sum"),
            demand_forecast=("demand_forecast", "sum"),
            demand_monthly=("demand_monthly", "sum"),
            replenishment_quantity=("replenishment_quantity", "sum"),
            aging_risk=("aging_risk", lambda s: s.mode().iloc[0] if len(s.mode()) else "<3 Months"),
        )
        agg["unit_cost"] = (agg["closing_stock_value"] / agg["closing_stock"].replace(0, pd.NA)).fillna(0.0)

        frac = WEEK_DAYS.get(week, 7) / 30.0
        base_demand = agg["demand_monthly"].where(agg["demand_monthly"] > 0, agg["demand_forecast"])
        agg["wk_demand"] = (base_demand * frac).round(0)
        agg["wk_cash"] = (agg["wk_demand"] * agg["unit_cost"]).round(2)
        agg["mo_cash"] = (agg["demand_forecast"] * agg["unit_cost"]).round(2)

        agg = agg.sort_values(["replenishment_quantity", "demand_forecast"], ascending=False).head(limit)

        rows: List[Dict[str, Any]] = []
        for _, r in agg.iterrows():
            rows.append({
                "Material": str(r["material_id"]),
                "Material Name": str(r["material_desc"]),
                "Order_Quantity": round(float(r["replenishment_quantity"]), 0),
                "Demand_Forecast": round(float(r["demand_forecast"]), 0),
                "Cash_Flow_Prediction": float(r["mo_cash"]),
                "Stock": round(float(r["closing_stock"]), 0),
                "Stock_Value": round(float(r["closing_stock_value"]), 2),
                "Aging": str(r["aging_risk"]),
                f"Demand_Forecast_{week}": float(r["wk_demand"]),
                f"Cash_Flow_Prediction_{week}": float(r["wk_cash"]),
            })

        return FilteredDataResponse(
            data=rows, week_filter=week, plant_filter=plant, total_records=len(rows),
        )
    except Exception as e:
        logging.exception("weekly forecast data error")
        raise HTTPException(status_code=500, detail=str(e))
