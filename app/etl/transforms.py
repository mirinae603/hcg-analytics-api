"""Compute per-KPI aggregate parquet tables from the curated facts.

Output columns are designed to map cleanly to the API JSON each endpoint
returns. Forecasting KPIs (D1-D8) are produced by app.forecast (Phase 4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

KPI = Path(settings.KPI_DIR)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

AGING_BINS = [-1, 30, 90, 180, 365, np.inf]
AGING_LABELS = ["0-30", "31-90", "91-180", "181-365", "365+"]


def _save(df: pd.DataFrame, name: str) -> None:
    KPI.mkdir(parents=True, exist_ok=True)
    df.to_parquet(KPI / f"{name}.parquet", index=False)
    print(f"[kpi] {name}: {len(df):,} rows")


def _aging_bucket(days: pd.Series) -> pd.Series:
    return pd.cut(days, bins=AGING_BINS, labels=AGING_LABELS)


def _attach_material_meta(df: pd.DataFrame, dim_material: pd.DataFrame) -> pd.DataFrame:
    meta = dim_material[["material", "material_desc", "material_group"]].drop_duplicates("material")
    return df.merge(meta, on="material", how="left", suffixes=("", "_m"))


# --------------------------------------------------------------------------- #
# INVENTORY
# --------------------------------------------------------------------------- #
def build_inventory_kpis(inv: pd.DataFrame, cons: pd.DataFrame, dim_material: pd.DataFrame):
    snapshot = inv["snapshot_date"].max()

    # ---- A1 Current Stock Value (per plant+material) ----
    a1 = (inv.groupby(["plant", "material", "material_desc", "material_group"], dropna=False)
              .agg(stock_qty=("qty", "sum"),
                   stock_value_cost=("total_cost", "sum"),
                   stock_value_mrp=("total_mrp", "sum"))
              .reset_index())
    _save(a1, "kpi_stock_value")

    # ---- A7 Aging Distribution (bucketed value/qty) ----
    inv2 = inv.assign(aging_bucket=_aging_bucket(inv["aging_days"]))
    a7 = (inv2.groupby(["plant", "material_group", "aging_bucket"], observed=True, dropna=False)
              .agg(stock_value=("total_cost", "sum"),
                   stock_qty=("qty", "sum"),
                   sku_count=("material", "nunique"))
              .reset_index())
    _save(a7, "kpi_aging_distribution")

    # ---- last sale / last purchase per material ----
    last_sale = cons.groupby("material")["posting_date"].max().rename("last_sale_date")
    last_purchase = inv.groupby("material")["grn_date"].max().rename("last_purchase_date")

    # ---- A2 Inventory Aging (per plant+material) ----
    grp = inv.groupby(["plant", "material", "material_desc", "material_group"], dropna=False)
    a2 = grp.agg(closing_stock_quantity=("qty", "sum"),
                 closing_stock_value=("total_cost", "sum"),
                 aging_days=("aging_days", lambda s: np.average(s, weights=inv.loc[s.index, "qty"].clip(lower=0.0001)))
                 ).reset_index()
    a2 = a2.merge(last_sale, on="material", how="left").merge(last_purchase, on="material", how="left")
    a2["age_since_last_sale_days"] = (snapshot - a2["last_sale_date"]).dt.days
    a2["age_since_last_purchase_days"] = (snapshot - a2["last_purchase_date"]).dt.days
    a2["aging_category"] = _aging_bucket(a2["aging_days"]).astype(str)
    _save(a2, "kpi_inventory_aging")

    # ---- A3 Days of Inventory on Hand ----
    days_span = max((cons["posting_date"].max() - cons["posting_date"].min()).days, 1)
    cons_qty = cons.groupby(["plant", "material"])["qty"].sum().rename("consumption_qty").reset_index()
    a3 = a1[["plant", "material", "material_desc", "material_group", "stock_qty"]].merge(
        cons_qty, on=["plant", "material"], how="left")
    a3["consumption_qty"] = a3["consumption_qty"].fillna(0)
    a3["avg_daily_consumption"] = a3["consumption_qty"] / days_span
    a3["doh_days"] = np.where(a3["avg_daily_consumption"] > 0,
                              a3["stock_qty"] / a3["avg_daily_consumption"], np.nan)
    _save(a3, "kpi_doh")

    # ---- A8 Inventory Health Score ----
    months = max(days_span / 30.0, 1)
    cons_cost = cons.groupby(["plant", "material"])["amount_lc"].sum().rename("consumption_cost").reset_index()
    a8 = a2[["plant", "material", "material_desc", "material_group",
             "closing_stock_value", "aging_days"]].merge(cons_cost, on=["plant", "material"], how="left")
    a8["consumption_cost"] = a8["consumption_cost"].fillna(0)
    a8["turnover_annualized"] = np.where(a8["closing_stock_value"] > 0,
                                         a8["consumption_cost"] / a8["closing_stock_value"] * (12 / months), 0)
    # sub-scores 0-100
    aging_score = (1 - (a8["aging_days"].clip(0, 365) / 365)) * 100
    turn_score = (a8["turnover_annualized"].clip(0, 6) / 6) * 100
    move_score = np.where(a8["consumption_cost"] > 0, 100, 0)
    a8["health_score"] = (0.4 * aging_score + 0.4 * turn_score + 0.2 * move_score).round(1)
    a8["health_tier"] = pd.cut(a8["health_score"], [-1, 40, 70, 101],
                               labels=["At Risk", "Watch", "Healthy"]).astype(str)
    _save(a8, "kpi_health_score")

    # ---- A9 Non-Moving Inventory (no consumption in window OR aging>180) ----
    a9 = a2.merge(last_sale, on="material", how="left", suffixes=("", "_ls"))
    consumed_materials = set(cons["material"].unique())
    a9["consumed_in_window"] = a9["material"].isin(consumed_materials)
    a9 = a9[(~a9["consumed_in_window"]) | (a9["aging_days"] > 180)].copy()
    a9["reason"] = np.where(~a9["consumed_in_window"], "No consumption in 6mo", "Aging > 180d")
    a9 = a9[["plant", "material", "material_desc", "material_group",
             "closing_stock_quantity", "closing_stock_value", "aging_days",
             "last_sale_date", "reason"]]
    _save(a9, "kpi_non_moving")

    # ---- A10 Inventory Risk Classification ----
    exp = inv.groupby(["plant", "material"])["expiry_date"].min().rename("nearest_expiry").reset_index()
    a10 = a2[["plant", "material", "material_desc", "material_group",
              "closing_stock_quantity", "closing_stock_value", "aging_days"]].merge(
        exp, on=["plant", "material"], how="left")
    a10["days_to_expiry"] = (a10["nearest_expiry"] - snapshot).dt.days
    a10["consumed"] = a10["material"].isin(consumed_materials)

    def _risk(r):
        if pd.notna(r["days_to_expiry"]) and r["days_to_expiry"] <= 90:
            return "High"
        if r["aging_days"] > 365 or not r["consumed"]:
            return "High"
        if r["aging_days"] > 180:
            return "Medium"
        return "Low"
    a10["risk_level"] = a10.apply(_risk, axis=1)
    _save(a10, "kpi_risk_classification")

    # ---- E1 Near Expiry (batch level) ----
    e1 = inv[inv["expiry_date"].notna()].copy()
    e1["days_to_expiry"] = (e1["expiry_date"] - snapshot).dt.days
    e1 = e1[e1["days_to_expiry"] <= 180]
    e1["expiry_bucket"] = pd.cut(e1["days_to_expiry"], [-9999, 0, 30, 90, 180],
                                 labels=["Expired", "0-30d", "31-90d", "91-180d"]).astype(str)
    e1 = e1[["plant", "material", "material_desc", "material_group", "batch",
             "expiry_date", "days_to_expiry", "expiry_bucket", "qty", "total_cost", "total_mrp"]]
    _save(e1, "kpi_near_expiry")


# --------------------------------------------------------------------------- #
# STOCK LEVEL CHANGE (A6) — GRN inflow minus consumption outflow per month
# --------------------------------------------------------------------------- #
def build_stock_change(grn: pd.DataFrame, cons: pd.DataFrame, dim_material: pd.DataFrame):
    inflow = (grn.groupby(["plant", "material", "year", "month"], dropna=False)["gr_qty"]
                 .sum().rename("inflow").reset_index())
    outflow = (cons.groupby(["plant", "material", "year", "month"], dropna=False)["qty"]
                  .sum().rename("outflow").reset_index())
    m = inflow.merge(outflow, on=["plant", "material", "year", "month"], how="outer")
    m[["inflow", "outflow"]] = m[["inflow", "outflow"]].fillna(0)
    m["stock_change"] = m["inflow"] - m["outflow"]
    m = _attach_material_meta(m, dim_material)
    m["material_group"] = m["material_group"].fillna("")
    out = m[["year", "month", "plant", "material", "material_desc", "material_group",
             "inflow", "outflow", "stock_change"]]
    _save(out, "kpi_stock_change")

    grp = (out.groupby(["year", "month", "plant", "material_group"], dropna=False)["stock_change"]
              .sum().reset_index())
    _save(grp, "kpi_stock_change_by_group")


# --------------------------------------------------------------------------- #
# PROCUREMENT
# --------------------------------------------------------------------------- #
def build_procurement_kpis(po: pd.DataFrame, dim_material: pd.DataFrame):
    po = po.copy()
    po["category"] = po["major_group"].replace({"nan": np.nan}).fillna("Uncategorized")

    # ---- B1 Purchase Value (vendor x category x month x plant) ----
    b1 = (po.groupby(["plant", "vendor_name", "category", "year", "month"], dropna=False)
             .agg(purchase_value=("total_value_wo_tax", "sum"),
                  purchase_qty=("po_qty", "sum"),
                  po_lines=("po_no", "count"))
             .reset_index())
    _save(b1, "kpi_purchase_value")

    # ---- B2 Monthly SKU Purchase Value ----
    b2 = (po.groupby(["plant", "material", "year", "month"], dropna=False)
             .agg(monthly_purchase_value=("total_value_wo_tax", "sum"),
                  purchase_qty=("po_qty", "sum"))
             .reset_index())
    b2 = _attach_material_meta(b2, dim_material)
    _save(b2[["year", "month", "plant", "material", "material_desc",
              "material_group", "monthly_purchase_value", "purchase_qty"]],
          "kpi_monthly_purchase_value")

    # ---- B3 Procurement Variance (plant monthly totals, MoM) ----
    monthly = (po.groupby(["plant", "year", "month"], dropna=False)["total_value_wo_tax"]
                  .sum().rename("purchase_value").reset_index())
    monthly["month_idx"] = monthly["month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    monthly = monthly.sort_values(["plant", "year", "month_idx"])
    monthly["prev_value"] = monthly.groupby("plant")["purchase_value"].shift(1)
    monthly["variance_abs"] = monthly["purchase_value"] - monthly["prev_value"]
    monthly["variance_pct"] = np.where(monthly["prev_value"].fillna(0) > 0,
                                       monthly["variance_abs"] / monthly["prev_value"] * 100, np.nan)
    _save(monthly.drop(columns=["month_idx"]), "kpi_procurement_variance")

    # ---- B4 Vendor Volume Contribution ----
    vend = (po.groupby(["plant", "vendor_name"], dropna=False)
               .agg(vendor_value=("total_value_wo_tax", "sum"),
                    vendor_qty=("po_qty", "sum"),
                    po_lines=("po_no", "count"))
               .reset_index())
    plant_tot = vend.groupby("plant")["vendor_value"].transform("sum")
    vend["value_share_pct"] = np.where(plant_tot > 0, vend["vendor_value"] / plant_tot * 100, 0)
    _save(vend, "kpi_vendor_volume")

    # ---- B7 Purchase Distribution by Location ----
    loc = (po.groupby(["plant"], dropna=False)
              .agg(purchase_value=("total_value_wo_tax", "sum"),
                   purchase_qty=("po_qty", "sum"),
                   vendor_count=("vendor_name", "nunique"),
                   po_lines=("po_no", "count"))
              .reset_index())
    _save(loc, "kpi_purchase_by_location")


# --------------------------------------------------------------------------- #
# CONSUMPTION
# --------------------------------------------------------------------------- #
def build_consumption_kpis(cons: pd.DataFrame, dim_material: pd.DataFrame, dim_costcenter: pd.DataFrame):
    # ---- C8 Units Consumed per SKU ----
    c8 = (cons.groupby(["plant", "material", "year", "month"], dropna=False)
             .agg(total_units=("qty", "sum"), consumption_cost=("amount_lc", "sum"))
             .reset_index())
    c8 = _attach_material_meta(c8, dim_material)
    _save(c8[["year", "month", "plant", "material", "material_desc",
              "material_group", "total_units", "consumption_cost"]],
          "kpi_units_consumed")

    # ---- C9 Consumption by Department (cost center) ----
    c9 = (cons.groupby(["plant", "cost_ctr", "year", "month"], dropna=False)
             .agg(consumption_qty=("qty", "sum"), consumption_cost=("amount_lc", "sum"))
             .reset_index())
    c9 = c9.merge(dim_costcenter, on="cost_ctr", how="left")
    c9["department_name"] = c9["department_name"].fillna(c9["cost_ctr"])
    _save(c9, "kpi_consumption_by_department")


# --------------------------------------------------------------------------- #
# ADDITIONAL: cycle time, lead time, fill rate
# --------------------------------------------------------------------------- #
def build_additional_kpis(grn: pd.DataFrame, po: pd.DataFrame):
    # Clip invalid lead-time anomalies: negative (GR before PO = data error) and
    # implausible outliers (> 365 days). Keeps cycle/lead-time KPIs trustworthy.
    grn = grn.copy()
    for c in ("po_to_gr_tat", "pr_to_gr_tat"):
        grn.loc[(grn[c] < 0) | (grn[c] > 365), c] = np.nan

    # ---- E2 Procurement Cycle Time ----
    e2 = (grn.groupby(["plant", "year", "month"], dropna=False)
             .agg(avg_po_to_gr_tat=("po_to_gr_tat", "mean"),
                  avg_pr_to_gr_tat=("pr_to_gr_tat", "mean"),
                  gr_lines=("gr_no", "count"))
             .reset_index())
    _save(e2, "kpi_cycle_time")

    # ---- E3 Vendor Lead Time ----
    e3 = (grn.groupby(["vendor_name"], dropna=False)
             .agg(avg_lead_time_days=("po_to_gr_tat", "mean"),
                  median_lead_time_days=("po_to_gr_tat", "median"),
                  gr_lines=("gr_no", "count"))
             .reset_index())
    e3 = e3[e3["gr_lines"] >= 3].sort_values("avg_lead_time_days")
    _save(e3, "kpi_vendor_lead_time")

    # ---- E4 Fill Rate (per plant; per vendor) ----
    fr_plant = (po.groupby(["plant"], dropna=False)
                   .agg(ordered_qty=("po_qty", "sum"), open_qty=("open_qty", "sum"))
                   .reset_index())
    fr_plant["fill_rate_pct"] = np.where(fr_plant["ordered_qty"] > 0,
                                         (1 - fr_plant["open_qty"] / fr_plant["ordered_qty"]) * 100, np.nan)
    _save(fr_plant, "kpi_fill_rate")


# --------------------------------------------------------------------------- #
def build_all(curated: dict):
    inv = curated["inventory"]
    cons = curated["consumption"]
    grn = curated["grn"]
    po = curated["po"]
    dim_material = curated["dim_material"]
    dim_costcenter = curated["dim_costcenter"]

    print("=== Building inventory KPIs ===")
    build_inventory_kpis(inv, cons, dim_material)
    print("=== Building stock change (A6) ===")
    build_stock_change(grn, cons, dim_material)
    print("=== Building procurement KPIs ===")
    build_procurement_kpis(po, dim_material)
    print("=== Building consumption KPIs ===")
    build_consumption_kpis(cons, dim_material, dim_costcenter)
    print("=== Building additional KPIs ===")
    build_additional_kpis(grn, po)

    # Forecasting KPIs (D1-D8) — Phase 4
    try:
        from app.forecast import engine
        print("=== Building forecast KPIs ===")
        engine.build_all(curated)
    except Exception as e:  # forecast module not present yet
        print(f"[kpi] forecast stage skipped: {e}")
