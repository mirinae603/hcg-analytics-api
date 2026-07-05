"""Simple statistical demand forecasting (Phase 4).

Per (plant, material) series of 6 monthly consumption points:
  - linear least-squares trend over the observed months (missing months = 0 demand),
    clipped at 0 — a robust, explainable stand-in for Holt's linear method on short
    history; collapses to the mean when the slope is ~0.
  - 95% band = forecast ± z * residual std.

Outputs (parquet in KPI_DIR) match the shapes the existing endpoints consume:
  - forecast_sales      -> /forecast/sales-forecast & /forecast/cashflow-forecast
  - replenishment       -> /inventory/replenishment-data & /api/data
  - kpi_fulfillment     -> D3
  - kpi_stock_radar     -> D5
  - kpi_aging_risk_forecast -> D6
Plus a backtest (last-month hold-out) for Forecast Accuracy %.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

KPI = Path(settings.KPI_DIR)
H = settings.FORECAST_HORIZON_MONTHS
Z = settings.FORECAST_Z

# chronological order of the 6 available months
HIST_ORDER = [(2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]
MONTH_NAME = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
              7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}


def _monthly_matrix(cons: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Pivot to (plant,material) x month matrix in chronological order, missing=0."""
    cons = cons.copy()
    cons["ym"] = list(zip(cons["year"].astype("Int64"), cons["month_num"].astype("Int64")))
    piv = (cons.pivot_table(index=["plant", "material"], columns="ym",
                            values=value_col, aggfunc="sum", fill_value=0.0))
    for ym in HIST_ORDER:
        if ym not in piv.columns:
            piv[ym] = 0.0
    piv = piv[HIST_ORDER]
    return piv, piv.to_numpy(dtype=float)


SLOPE_DAMP = 0.5   # damp the linear trend to curb SKU-level overshoot
MA_WINDOW = 3      # moving-average baseline window


def _linear_forecast(M: np.ndarray, horizon: int):
    """Vectorized damped-trend + moving-average blend per row.

    Short, intermittent SKU series make a raw linear trend overshoot; we anchor the
    forecast on the trailing moving-average level and add a damped slope, which is a
    far more stable estimator on 6 points. Returns (fc, lo, hi, resid_std)."""
    n_rows, n = M.shape
    x = np.arange(n, dtype=float)
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()
    ybar = M.mean(axis=1)
    sxy = ((M - ybar[:, None]) * (x - xbar)[None, :]).sum(axis=1)
    slope = (sxy / sxx) * SLOPE_DAMP
    # baseline level = mean of trailing MA_WINDOW months
    level = M[:, -MA_WINDOW:].mean(axis=1)

    # in-sample fitted (level carried + damped slope around last point) for residuals
    fitted = level[:, None] + slope[:, None] * (x - (n - 1))[None, :]
    resid = M - fitted
    resid_std = resid.std(axis=1, ddof=0)

    step = np.arange(1, horizon + 1, dtype=float)
    fc = level[:, None] + slope[:, None] * step[None, :]
    fc = np.clip(fc, 0, None)
    band = Z * resid_std[:, None]
    lo = np.clip(fc - band, 0, None)
    hi = fc + band
    return fc, lo, hi, resid_std


def build_forecast_tables(cons: pd.DataFrame, inv: pd.DataFrame, dim_material: pd.DataFrame):
    qty_piv, Mq = _monthly_matrix(cons, "qty")
    cost_piv, Mc = _monthly_matrix(cons, "amount_lc")

    fc_q, lo_q, hi_q, _ = _linear_forecast(Mq, H)
    fc_c, lo_c, hi_c, _ = _linear_forecast(Mc, H)

    keys = qty_piv.index.to_frame(index=False)  # plant, material
    meta = dim_material[["material", "material_group"]].drop_duplicates("material")
    keys = keys.merge(meta, on="material", how="left")
    keys["material_group"] = keys["material_group"].fillna("")

    future = [(2026, 6 + i) if (6 + i) <= 12 else (2027, 6 + i - 12) for i in range(H)]
    rows = []
    # history rows (actuals)
    for j, (yy, mm) in enumerate(HIST_ORDER):
        d = f"{yy:04d}-{mm:02d}-01"
        rows.append(pd.DataFrame({
            "material_id": keys["material"].values,
            "plant": keys["plant"].values,
            "material_group": keys["material_group"].values,
            "posting_date": d,
            "sales_quantity": Mq[:, j],
            "sales_value": Mc[:, j],
            "sales_quantity_forecast": np.nan, "lower_bound_sales_quantity_forecast": np.nan,
            "upper_bound_sales_quantity_forecast": np.nan,
            "cashflow_forecast": np.nan, "lower_bound_cashflow_forecast": np.nan,
            "upper_bound_cashflow_forecast": np.nan,
        }))
    # forecast rows
    for j, (yy, mm) in enumerate(future):
        d = f"{yy:04d}-{mm:02d}-01"
        rows.append(pd.DataFrame({
            "material_id": keys["material"].values,
            "plant": keys["plant"].values,
            "material_group": keys["material_group"].values,
            "posting_date": d,
            "sales_quantity": np.nan, "sales_value": np.nan,
            "sales_quantity_forecast": fc_q[:, j],
            "lower_bound_sales_quantity_forecast": lo_q[:, j],
            "upper_bound_sales_quantity_forecast": hi_q[:, j],
            "cashflow_forecast": fc_c[:, j],
            "lower_bound_cashflow_forecast": lo_c[:, j],
            "upper_bound_cashflow_forecast": hi_c[:, j],
        }))
    forecast_sales = pd.concat(rows, ignore_index=True)
    forecast_sales.to_parquet(KPI / "forecast_sales.parquet", index=False)
    print(f"[forecast] forecast_sales: {len(forecast_sales):,} rows")

    # ---- next-month demand (sum of horizon) for replenishment ----
    demand_next = pd.DataFrame({
        "plant": keys["plant"].values,
        "material": keys["material"].values,
        "material_group": keys["material_group"].values,
        "demand_forecast": fc_q.sum(axis=1),         # horizon-month demand
        "demand_monthly": fc_q[:, 0],
    })
    return forecast_sales, demand_next


def build_replenishment(demand_next: pd.DataFrame, inv: pd.DataFrame, cons: pd.DataFrame,
                        vendor_lead: pd.DataFrame, dim_material: pd.DataFrame):
    stock = (inv.groupby(["plant", "material"])
                .agg(closing_stock=("qty", "sum"),
                     closing_stock_value=("total_cost", "sum"),
                     aging_days=("aging_days", "mean"),
                     unit_cost=("moving_avg_price", "mean"))
                .reset_index())
    df = stock.merge(demand_next, on=["plant", "material"], how="outer")
    df[["closing_stock", "closing_stock_value", "demand_forecast", "demand_monthly", "unit_cost"]] = \
        df[["closing_stock", "closing_stock_value", "demand_forecast", "demand_monthly", "unit_cost"]].fillna(0)

    # demand volatility for safety stock
    vol = (cons.groupby(["plant", "material"])["qty"]
              .apply(lambda s: float(np.std(s.values, ddof=0))).rename("demand_std").reset_index())
    df = df.merge(vol, on=["plant", "material"], how="left")
    df["demand_std"] = df["demand_std"].fillna(0)
    # avg lead time (months) ~ overall mean if vendor unknown
    lead_months = 0.25  # ~7.5 days default
    df["safe_stock"] = (Z * df["demand_std"] * np.sqrt(max(lead_months, 0.1))).round(2)

    df["replenishment_quantity"] = np.clip(df["demand_forecast"] + df["safe_stock"] - df["closing_stock"], 0, None)
    df["inventory_aging_risk"] = df["aging_days"] > 180
    df["aging_risk"] = pd.cut(df["aging_days"].fillna(0), [-1, 90, 180, 365, np.inf],
                              labels=["<3 Months", "3-6 Months", "6-12 Months", "1+ Year"]).astype(str)
    meta = dim_material[["material", "material_desc"]].drop_duplicates("material")
    df = df.merge(meta, on="material", how="left")
    df = df.rename(columns={"material": "material_id"})
    df.to_parquet(KPI / "stock_replenishment_and_aging_risk.parquet", index=False)
    print(f"[forecast] replenishment: {len(df):,} rows "
          f"({int((df['replenishment_quantity']>0).sum()):,} need replenishment)")

    # ---- D3 Fulfillment Rate ----
    ful = df[["plant", "material_id", "material_desc", "closing_stock", "demand_monthly"]].copy()
    ful["fulfillment_rate"] = np.where(ful["demand_monthly"] > 0,
                                       np.clip(ful["closing_stock"] / ful["demand_monthly"], 0, 1) * 100, 100.0)
    ful["coverage_months"] = np.where(ful["demand_monthly"] > 0,
                                      ful["closing_stock"] / ful["demand_monthly"], np.nan)
    ful.to_parquet(KPI / "kpi_fulfillment.parquet", index=False)
    print(f"[forecast] kpi_fulfillment: {len(ful):,} rows")

    # ---- D5 Stock Radar ----
    radar = df[["plant", "material_id", "material_desc", "material_group", "closing_stock",
                "demand_forecast", "aging_days", "aging_risk", "replenishment_quantity"]].copy()
    radar["coverage_months"] = np.where(radar["demand_forecast"] > 0,
                                        radar["closing_stock"] / (radar["demand_forecast"] / max(H, 1)), np.nan)

    def _radar(r):
        if r["replenishment_quantity"] > 0 and (pd.isna(r["coverage_months"]) or r["coverage_months"] < 1):
            return "Stock-Out Risk"
        if r["aging_days"] > 180:
            return "Overstock/Aging"
        return "Healthy"
    radar["radar_status"] = radar.apply(_radar, axis=1)
    radar.to_parquet(KPI / "kpi_stock_radar.parquet", index=False)
    print(f"[forecast] kpi_stock_radar: {len(radar):,} rows")

    # ---- D6 Aging Risk Forecast ----
    arf = df[["plant", "material_id", "material_desc", "material_group", "closing_stock",
              "demand_monthly", "aging_days"]].copy()
    # projected aging in 3 months if consumption continues at forecast pace
    arf["projected_aging_days"] = arf["aging_days"] + 90
    arf["will_stagnate"] = (arf["demand_monthly"] < (arf["closing_stock"] * 0.05))  # <5% monthly turn
    arf["aging_risk_forecast"] = np.where(
        arf["will_stagnate"] & (arf["closing_stock"] > 0), "Rising", "Stable")
    arf.to_parquet(KPI / "kpi_aging_risk_forecast.parquet", index=False)
    print(f"[forecast] kpi_aging_risk_forecast: {len(arf):,} rows")


def build_backtest(cons: pd.DataFrame):
    """Hold out the last month, forecast it, report accuracy at several aggregations.

    SKU-level demand is intermittent, so unweighted SKU MAPE is uninformative. The
    meaningful headline is the volume-weighted accuracy (large SKUs dominate spend)
    and the portfolio-aggregate accuracy (total demand), both reported here."""
    qty_piv, Mq = _monthly_matrix(cons, "qty")
    train = Mq[:, :-1]
    actual = Mq[:, -1]
    fc, _, _, _ = _linear_forecast(train, 1)
    pred = fc[:, 0]

    mask = actual > 0
    sku_mape = float(np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100) if mask.any() else None
    # volume-weighted MAPE (weight by actual units)
    w = actual[mask]
    wmape = float(np.sum(np.abs(actual[mask] - pred[mask])) / np.sum(w) * 100) if mask.any() else None
    # portfolio-aggregate (total demand) accuracy
    agg_err = abs(actual.sum() - pred.sum()) / actual.sum() * 100 if actual.sum() else None
    mae = float(np.mean(np.abs(actual - pred)))

    def acc(m):
        return round(max(0.0, 100 - m), 1) if m is not None else None

    bt = pd.DataFrame([
        {"metric": "Weighted Forecast Accuracy %", "value": acc(wmape)},
        {"metric": "Aggregate Forecast Accuracy %", "value": acc(agg_err)},
        {"metric": "SKU MAPE %", "value": round(sku_mape, 1) if sku_mape else None},
        {"metric": "Weighted MAPE %", "value": round(wmape, 1) if wmape else None},
        {"metric": "MAE (units)", "value": round(mae, 2)},
        {"metric": "Series count", "value": int(len(actual))},
    ])
    bt.to_parquet(KPI / "forecast_accuracy.parquet", index=False)
    print(f"[forecast] backtest: weighted-acc={acc(wmape)}% aggregate-acc={acc(agg_err)}% "
          f"SKU-MAPE={round(sku_mape,1) if sku_mape else None}%")


def build_all(curated: dict):
    cons = curated["consumption"]
    inv = curated["inventory"]
    dim_material = curated["dim_material"]
    vendor_lead = None
    _, demand_next = build_forecast_tables(cons, inv, dim_material)
    build_replenishment(demand_next, inv, cons, vendor_lead, dim_material)
    build_backtest(cons)
