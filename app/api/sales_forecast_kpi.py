"""Sales-quantity & cash-flow demand forecast endpoints (D2, D7).

Reads the ETL-produced `forecast_sales` parquet (history actuals + forward forecast
with 95% bounds). Output shapes are unchanged from the original CSV-backed version so
the existing frontend forecast pages render without changes.
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from typing import Optional, List, Dict, Any
import pandas as pd

from app.core import data_access as da

router = APIRouter()


def _series(plant: str, material: Optional[str], material_group: Optional[str],
            qty_col: str, lo_col: str, fc_col: str, hi_col: str) -> List[Dict[str, Any]]:
    df = da.load("forecast_sales").copy()
    df = da.filter_plant(df, plant)
    if material and material != "All Items":
        df = df[df["material_id"].astype(str) == str(material)]
    elif material_group:
        df = df[df["material_group"].astype(str) == str(material_group)]
    if df.empty:
        return []
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    g = (df.groupby("posting_date")
           .agg(actual=(qty_col, "sum"), lo=(lo_col, "sum"),
                fc=(fc_col, "sum"), hi=(hi_col, "sum"))
           .reset_index().sort_values("posting_date"))
    out = []
    for _, r in g.iterrows():
        out.append({
            "Plant": plant,
            "Posting Date": r["posting_date"].strftime("%m/%d/%y"),
            "sales_qty": None if pd.isna(r["actual"]) else float(r["actual"]),
            "Lower_Bound_Sales_Quantity_Forecast": None if pd.isna(r["lo"]) else float(r["lo"]),
            "Sales_Quantity_Forecast": None if pd.isna(r["fc"]) else float(r["fc"]),
            "Upper_Bound_Sales_Quantity_Forecast": None if pd.isna(r["hi"]) else float(r["hi"]),
        })
    return out


@router.get("/forecast/sales-forecast")
def sales_forecast(Plant: str = Query(...), Material: Optional[str] = Query(None),
                   MaterialGroup: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    return _series(Plant, Material, MaterialGroup,
                   "sales_quantity", "lower_bound_sales_quantity_forecast",
                   "sales_quantity_forecast", "upper_bound_sales_quantity_forecast")


@router.get("/forecast/cashflow-forecast")
def cashflow_forecast(Plant: str = Query(...), Material: Optional[str] = Query(None),
                      MaterialGroup: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    return _series(Plant, Material, MaterialGroup,
                   "sales_value", "lower_bound_cashflow_forecast",
                   "cashflow_forecast", "upper_bound_cashflow_forecast")


@router.get("/forecast/accuracy")
def forecast_accuracy():
    return da._clean_records(da.load("forecast_accuracy"))
