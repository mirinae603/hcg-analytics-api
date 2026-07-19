"""
HCG AI Analyst — deterministic analytics semantic layer.

This is the *tool* layer the LLM orchestrator calls. Every function reads the real
parquet aggregates and returns a uniform, tabular Result — NO SQL DB, NO
LLM-generated code execution. The model only chooses *which* function and *which*
params; the numbers are always computed here, so answers can never be fabricated.

Result shape (dict):
  title:  str                      human title for the cut
  rows:   list[dict]               the actual data rows
  columns:list[{key,label,kind}]   kind ∈ inr|pct|num|days|text  (for formatting)
  stats:  dict                     headline scalars (totals etc.)
  suggested_chart: dict|None       {type,x,y,series,title}  type ∈ bar|line|pie|area
  note:   str                      methodology / caveat surfaced to the user
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from app.core import data_access as da

_KPI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kpi")


# ───────────────────────── helpers ─────────────────────────
def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_KPI, name + ".parquet"))


def _clean_group(g) -> str:
    g = str(g).strip()
    if not g or g.lower() in ("nan", "none"):
        return "Uncategorised"
    g = g.split("-", 1)[-1] if "-" in g else g
    return g.strip().title() or "Uncategorised"


_MONTH_ORDER = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
_MONTH_LABEL = {"2025-12": "Dec 25", "2026-01": "Jan 26", "2026-02": "Feb 26",
                "2026-03": "Mar 26", "2026-04": "Apr 26", "2026-05": "May 26"}


def _result(title, rows, columns, stats=None, chart=None, note=""):
    return {"title": title, "rows": rows, "columns": columns,
            "stats": stats or {}, "suggested_chart": chart, "note": note}


def _plant_filter(df, plant):
    if plant and str(plant).lower() not in ("all", "all plants", "") and "plant" in df.columns:
        return da.filter_plant(df, da.resolve_plant(plant))
    return df


# ───────────────────────── REVENUE & MARGIN ─────────────────────────
def revenue(dimension: str = "manufacturer", metric: str = "revenue", top_n: int = 10):
    """Billed IP+OP pharmacy revenue / true margin (MRP − cost).
    dimension: manufacturer | hospital | category | material | month
    metric:    revenue | margin | margin_pct | qty
    """
    dimension = (dimension or "manufacturer").lower().strip()
    metric = (metric or "revenue").lower().strip()
    top_n = int(top_n or 10)

    if dimension == "month":
        m = _load("sales_monthly").groupby("month", as_index=False)[["revenue", "cost", "qty"]].sum()
        m["margin"] = m["revenue"] - m["cost"]
        m["margin_pct"] = np.where(m["revenue"] > 0, m["margin"] / m["revenue"] * 100, 0.0)
        m["__k"] = m["month"].map({v: i for i, v in enumerate(_MONTH_ORDER)})
        m = m.sort_values("__k")
        rows = [{"name": _MONTH_LABEL.get(r["month"], r["month"]), "revenue": float(r["revenue"]),
                 "margin": float(r["margin"]), "margin_pct": float(r["margin_pct"]), "qty": float(r["qty"])}
                for _, r in m.iterrows()]
        chart = {"type": "line", "x": "name", "y": metric if metric in ("revenue", "margin", "margin_pct", "qty") else "revenue",
                 "title": "Revenue by month"}
        stats = {"total_revenue": float(m["revenue"].sum()), "total_margin": float(m["margin"].sum())}
        return _result("Revenue by month", rows, _rev_cols(), stats, chart,
                       "Billed IP+OP pharmacy revenue over the 6-month window.")

    src = {"manufacturer": "sales_by_manufacturer", "hospital": "sales_by_hospital",
           "material": "sales_by_material", "category": "sales_by_material"}.get(dimension, "sales_by_manufacturer")
    df = _load(src).copy()
    if dimension == "category":
        df["name"] = df["group"].map(_clean_group)
        g = df.groupby("name", as_index=False)[["revenue", "cost", "qty"]].sum()
    elif dimension == "material":
        g = df.rename(columns={"desc": "name"})[["name", "revenue", "cost", "qty"]].copy()
    else:
        namecol = "manufacturer" if dimension == "manufacturer" else "hospital"
        g = df.rename(columns={namecol: "name"})[["name", "revenue", "cost", "qty"]].copy()
    g["margin"] = g["revenue"] - g["cost"]
    g["margin_pct"] = np.where(g["revenue"] > 0, g["margin"] / g["revenue"] * 100, 0.0)
    sortcol = metric if metric in ("revenue", "margin", "margin_pct", "qty") else "revenue"
    g = g.sort_values(sortcol, ascending=False).head(top_n)
    rows = [{"name": str(r["name"]), "revenue": float(r["revenue"]), "margin": float(r["margin"]),
             "margin_pct": float(r["margin_pct"]), "qty": float(r["qty"])} for _, r in g.iterrows()]
    title = f"Top {len(rows)} {dimension} by {sortcol.replace('_', ' ')}"
    chart = {"type": "bar", "x": "name", "y": sortcol, "title": title}
    stats = {"total_revenue": float(_load(src)["revenue"].sum())}
    return _result(title, rows, _rev_cols(), stats, chart,
                   "Margin = billed MRP − actual cost. IP+OP, 6-month window.")


def _rev_cols():
    return [{"key": "name", "label": "Name", "kind": "text"},
            {"key": "revenue", "label": "Revenue", "kind": "inr"},
            {"key": "margin", "label": "Margin", "kind": "inr"},
            {"key": "margin_pct", "label": "Margin %", "kind": "pct"},
            {"key": "qty", "label": "Qty", "kind": "num"}]


# ───────────────────────── PROCUREMENT ─────────────────────────
def procurement(view: str = "spend", dimension: str = "vendor", top_n: int = 10):
    """Procurement analytics.
    view: spend (by vendor|category|location|month) | vendors | open_po | savings | monthly
    """
    view = (view or "spend").lower().strip()
    dimension = (dimension or "vendor").lower().strip()
    top_n = int(top_n or 10)

    if view in ("vendors", "spend") and dimension == "vendor":
        vv = _load("kpi_vendor_volume")
        total = float(vv["vendor_value"].sum())
        # aggregate across plants — the KPI table has one row per vendor×plant
        v = vv.groupby("vendor_name", as_index=False).agg(vendor_value=("vendor_value", "sum"), po_lines=("po_lines", "sum"))
        v["share_pct"] = np.where(total > 0, v["vendor_value"] / total * 100, 0.0)
        v = v.sort_values("vendor_value", ascending=False).head(top_n)
        rows = [{"name": str(r["vendor_name"]), "spend": float(r["vendor_value"]),
                 "share_pct": float(r["share_pct"]), "po_lines": int(r["po_lines"])} for _, r in v.iterrows()]
        cols = [{"key": "name", "label": "Vendor", "kind": "text"}, {"key": "spend", "label": "Spend", "kind": "inr"},
                {"key": "share_pct", "label": "Share %", "kind": "pct"}, {"key": "po_lines", "label": "PO lines", "kind": "num"}]
        chart = {"type": "bar", "x": "name", "y": "spend", "title": "Top vendors by spend"}
        return _result(f"Top {len(rows)} vendors by spend", rows, cols, {"total_spend": total}, chart)

    if view == "spend" and dimension in ("category", "location", "month"):
        pv = _load("kpi_purchase_value")
        if dimension == "category":
            pv = pv.copy(); pv["name"] = pv["category"].map(_clean_group)
            g = pv.groupby("name", as_index=False)["purchase_value"].sum().sort_values("purchase_value", ascending=False).head(top_n)
            chart_type = "bar"
        elif dimension == "location":
            loc = _load("kpi_purchase_by_location").rename(columns={"plant": "name"})
            g = loc[["name", "purchase_value"]].sort_values("purchase_value", ascending=False).head(top_n)
            chart_type = "bar"
        else:  # month
            pv = pv.copy(); pv["ym"] = pv["year"].astype(str) + "-" + pv["month"].astype(str).str.zfill(2)
            g = pv.groupby("ym", as_index=False)["purchase_value"].sum()
            g["__k"] = g["ym"].map({v: i for i, v in enumerate(_MONTH_ORDER)})
            g = g.sort_values("__k"); g["name"] = g["ym"].map(lambda x: _MONTH_LABEL.get(x, x))
            chart_type = "line"
        rows = [{"name": str(r["name"]), "spend": float(r["purchase_value"])} for _, r in g.iterrows()]
        cols = [{"key": "name", "label": dimension.title(), "kind": "text"}, {"key": "spend", "label": "Spend", "kind": "inr"}]
        chart = {"type": chart_type, "x": "name", "y": "spend", "title": f"Purchase spend by {dimension}"}
        return _result(f"Purchase spend by {dimension}", rows, cols,
                       {"total_spend": float(_load("kpi_purchase_value")["purchase_value"].sum())}, chart)

    if view == "open_po":
        po = _load("kpi_purchase_value")  # fallback; real open-PO comes from fact_po
        try:
            fp = da.load("fact_po")
            op = fp[fp["open_qty"] > 0].copy()
            op["open_value"] = op["open_qty"] * op["net_price"]
            op["name"] = op["major_group"].map(_clean_group)
            g = op.groupby("name", as_index=False).agg(open_value=("open_value", "sum"), pos=("po_no", "nunique"))
            g = g.sort_values("open_value", ascending=False).head(top_n)
            rows = [{"name": str(r["name"]), "open_value": float(r["open_value"]), "open_pos": int(r["pos"])} for _, r in g.iterrows()]
            cols = [{"key": "name", "label": "Category", "kind": "text"}, {"key": "open_value", "label": "Open value", "kind": "inr"},
                    {"key": "open_pos", "label": "Open POs", "kind": "num"}]
            chart = {"type": "bar", "x": "name", "y": "open_value", "title": "Open POs by category"}
            return _result("Open purchase orders by category", rows, cols,
                           {"total_open_value": float(op["open_value"].sum()), "total_open_pos": int(op["po_no"].nunique())}, chart,
                           "Undelivered order value = open qty × net price.")
        except Exception:
            pass

    if view == "savings":
        try:
            g = da.load("fact_grn")
            d = g[(g["net_price"] > 0) & (g["gr_qty"] > 0)][["material", "material_desc", "net_price", "gr_qty"]].copy()
            st = d.groupby("material")["net_price"].agg(["min", "max", "median", "size"])
            st = st[(st["size"] >= 4) & ((st["max"] / st["min"].replace(0, np.nan)) <= 2.5)].dropna(subset=["max"])
            d2 = d[d["material"].isin(st.index)].merge(st[["median"]], left_on="material", right_index=True)
            d2["over"] = (d2["net_price"] - d2["median"]).clip(lower=0) * d2["gr_qty"]
            opp = d2.groupby(["material", "material_desc"], as_index=False).agg(over=("over", "sum"), med=("median", "first"), pmax=("net_price", "max"))
            opp = opp[opp["over"] > 0].sort_values("over", ascending=False).head(top_n)
            rows = [{"name": str(r["material_desc"]), "overpay": float(r["over"]),
                     "median_price": float(r["med"]), "max_price": float(r["pmax"])} for _, r in opp.iterrows()]
            cols = [{"key": "name", "label": "Item", "kind": "text"}, {"key": "overpay", "label": "Overpay", "kind": "inr"},
                    {"key": "median_price", "label": "Median price", "kind": "inr"}, {"key": "max_price", "label": "Highest price", "kind": "inr"}]
            chart = {"type": "bar", "x": "name", "y": "overpay", "title": "Price consolidation opportunity"}
            return _result("Price consolidation opportunity", rows, cols, {}, chart,
                           "Overpay vs each item's own median price (same-unit buys, ≥4 purchases). Negotiation headroom, not guaranteed saving.")
        except Exception:
            pass

    # default → vendor spend
    return procurement("vendors", "vendor", top_n)


# ───────────────────────── INVENTORY ─────────────────────────
def inventory(view: str = "stock_value", top_n: int = 10, plant: str = None):
    """Inventory analytics.
    view: stock_value | doh | aging | non_moving | health | expiry
    """
    view = (view or "stock_value").lower().strip()
    top_n = int(top_n or 10)

    if view == "stock_value":
        sv = _plant_filter(_load("kpi_stock_value"), plant).copy()
        sv["name"] = sv["material_group"].map(_clean_group)
        g = sv.groupby("name", as_index=False).agg(value=("stock_value_cost", "sum"), skus=("material", "nunique"))
        g = g.sort_values("value", ascending=False).head(top_n)
        rows = [{"name": str(r["name"]), "value": float(r["value"]), "skus": int(r["skus"])} for _, r in g.iterrows()]
        cols = [{"key": "name", "label": "Category", "kind": "text"}, {"key": "value", "label": "Stock value", "kind": "inr"},
                {"key": "skus", "label": "SKUs", "kind": "num"}]
        chart = {"type": "bar", "x": "name", "y": "value", "title": "Stock value by category"}
        return _result("Stock value by category", rows, cols,
                       {"total_stock_value": float(sv["stock_value_cost"].sum())}, chart, "Closing stock at cost.")

    if view == "doh":
        doh = _plant_filter(_load("kpi_doh"), plant).copy()
        moving = doh[doh["doh_days"].notna() & (doh["doh_days"] > 0)]
        median = float(moving["doh_days"].median()) if len(moving) else 0.0
        doh["name"] = doh["material_group"].map(_clean_group)
        g = doh.groupby("name", as_index=False).agg(qty=("stock_qty", "sum"), daily=("avg_daily_consumption", "sum"))
        g["doh"] = np.where(g["daily"] > 0, g["qty"] / g["daily"], np.nan)
        g = g.dropna(subset=["doh"]).sort_values("doh", ascending=False).head(top_n)
        rows = [{"name": str(r["name"]), "doh_days": round(float(r["doh"]), 1)} for _, r in g.iterrows()]
        cols = [{"key": "name", "label": "Category", "kind": "text"}, {"key": "doh_days", "label": "Days of cover", "kind": "days"}]
        chart = {"type": "bar", "x": "name", "y": "doh_days", "title": "Days of cover by category"}
        return _result("Days of inventory on hand", rows, cols,
                       {"median_doh_days": round(median, 1), "moving_skus": int(len(moving))}, chart,
                       f"Portfolio median days of cover = {median:.0f}d across moving SKUs (median, not mean — the tail of overstock skews the mean).")

    if view == "aging":
        ag = _plant_filter(_load("kpi_aging_distribution"), plant).copy()
        g = ag.groupby("aging_bucket", as_index=False).agg(value=("stock_value", "sum"), skus=("sku_count", "sum"))
        order = ["0-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days", "> 1 Year"]
        g["__k"] = g["aging_bucket"].map(lambda b: next((i for i, o in enumerate(order) if o.lower() in str(b).lower()), 99))
        g = g.sort_values("__k")
        rows = [{"name": str(r["aging_bucket"]), "value": float(r["value"]), "skus": int(r["skus"])} for _, r in g.iterrows()]
        cols = [{"key": "name", "label": "Age", "kind": "text"}, {"key": "value", "label": "Stock value", "kind": "inr"},
                {"key": "skus", "label": "SKUs", "kind": "num"}]
        chart = {"type": "bar", "x": "name", "y": "value", "title": "Inventory aging distribution"}
        return _result("Inventory aging distribution", rows, cols, {"total_value": float(g["value"].sum())}, chart)

    if view == "non_moving":
        nm = _plant_filter(_load("kpi_non_moving"), plant).copy()
        nm = nm.sort_values("closing_stock_value", ascending=False).head(top_n)
        rows = [{"name": str(r["material_desc"]), "value": float(r["closing_stock_value"]),
                 "aging_days": int(r["aging_days"]) if pd.notna(r["aging_days"]) else None} for _, r in nm.iterrows()]
        cols = [{"key": "name", "label": "Item", "kind": "text"}, {"key": "value", "label": "Stuck value", "kind": "inr"},
                {"key": "aging_days", "label": "Age", "kind": "days"}]
        chart = {"type": "bar", "x": "name", "y": "value", "title": "Non-moving stock by value"}
        return _result("Non-moving inventory", rows, cols, {}, chart, "Items with no recent movement, ranked by stuck capital.")

    if view == "health":
        hs = _plant_filter(_load("kpi_health_score"), plant).copy()
        g = hs.groupby("health_tier", as_index=False).agg(skus=("material", "nunique"), value=("closing_stock_value", "sum"))
        rows = [{"name": str(r["health_tier"]), "skus": int(r["skus"]), "value": float(r["value"])} for _, r in g.iterrows()]
        cols = [{"key": "name", "label": "Health tier", "kind": "text"}, {"key": "skus", "label": "SKUs", "kind": "num"},
                {"key": "value", "label": "Stock value", "kind": "inr"}]
        chart = {"type": "pie", "x": "name", "y": "value", "title": "Inventory health mix"}
        return _result("Inventory health score mix", rows, cols, {}, chart)

    if view == "expiry":
        return expiry()

    return inventory("stock_value", top_n, plant)


# ───────────────────────── EXPIRY ─────────────────────────
_EXP_ORDER = ["Expired", "0-30d", "31-90d", "91-180d", "181-365d", "365d+"]


def expiry(slab: str = None, top_n: int = 15, plant: str = None):
    """Near-expiry ladder (full 6 bands from fact_inventory vs snapshot), or the item list for one band.
    slab: None → ladder ; else one of Expired|0-30d|31-90d|91-180d|181-365d|365d+
    """
    df = _plant_filter(da.load("fact_inventory"), plant).copy()
    df = df.dropna(subset=["expiry_date"])
    df = df[df["qty"] > 0]
    snap = pd.to_datetime(df["snapshot_date"], errors="coerce").max()
    if pd.isna(snap):
        snap = pd.Timestamp("2026-05-31")
    df["dte"] = (pd.to_datetime(df["expiry_date"], errors="coerce") - snap).dt.days
    df = df.dropna(subset=["dte"])
    bins = [-10**12, -1, 30, 90, 180, 365, 10**12]
    df["slab"] = pd.cut(df["dte"], bins=bins, labels=_EXP_ORDER, right=True)

    if slab:
        sub = df[df["slab"].astype(str) == slab].sort_values("total_cost", ascending=False).head(int(top_n))
        rows = [{"name": str(r["material_desc"]), "value": float(r["total_cost"]),
                 "days_left": int(r["dte"]), "expiry": pd.to_datetime(r["expiry_date"]).strftime("%d %b %Y")} for _, r in sub.iterrows()]
        cols = [{"key": "name", "label": "Item", "kind": "text"}, {"key": "value", "label": "Value", "kind": "inr"},
                {"key": "days_left", "label": "Days left", "kind": "days"}, {"key": "expiry", "label": "Expiry", "kind": "text"}]
        chart = {"type": "bar", "x": "name", "y": "value", "title": f"{slab} — items by value"}
        return _result(f"Near-expiry items — {slab}", rows, cols,
                       {"band_value": float(df[df['slab'].astype(str) == slab]["total_cost"].sum())}, chart)

    rows = []
    for s in _EXP_ORDER:
        seg = df[df["slab"] == s]
        rows.append({"name": s, "value": float(seg["total_cost"].sum()),
                     "items": int(seg["material"].nunique()) if len(seg) else 0})
    cols = [{"key": "name", "label": "Band", "kind": "text"}, {"key": "value", "label": "Value", "kind": "inr"},
            {"key": "items", "label": "Items", "kind": "num"}]
    chart = {"type": "bar", "x": "name", "y": "value", "title": "Expiry ladder by cost value"}
    actionable = sum(r["value"] for r in rows if r["name"] in ("Expired", "0-30d", "31-90d", "91-180d"))
    return _result("Expiry ladder", rows, cols,
                   {"actionable_within_180d": actionable, "as_on": snap.strftime("%d %b %Y")}, chart,
                   f"As on {snap.strftime('%d %b %Y')}. ₹{actionable/1e7:.2f}Cr expires within 180 days.")


# ───────────────────────── STOCK RISK ─────────────────────────
def stock_risk(status: str = None, top_n: int = 15, plant: str = None):
    """Replenishment & stock-out risk. status: stock-out | reorder | overstock | None (summary)."""
    df = _plant_filter(da.load("stock_replenishment_and_aging_risk"), plant).copy()
    status = (status or "").lower().strip()

    if status in ("stock-out", "stockout", "reorder", "order"):
        risk = df[df["replenishment_quantity"] > 0].sort_values("replenishment_quantity", ascending=False).head(int(top_n))
        rows = [{"name": str(r["material_desc"]), "reorder_qty": float(r["replenishment_quantity"]),
                 "on_hand": float(r["closing_stock"]), "monthly_demand": float(r["demand_monthly"])} for _, r in risk.iterrows()]
        cols = [{"key": "name", "label": "Item", "kind": "text"}, {"key": "reorder_qty", "label": "Reorder qty", "kind": "num"},
                {"key": "on_hand", "label": "On hand", "kind": "num"}, {"key": "monthly_demand", "label": "Monthly demand", "kind": "num"}]
        chart = {"type": "bar", "x": "name", "y": "reorder_qty", "title": "Top reorder needs"}
        return _result("Items to reorder", rows, cols, {"items_to_reorder": int((df['replenishment_quantity'] > 0).sum())}, chart)

    # summary — risk band counts
    g = df.groupby("aging_risk", as_index=False).agg(skus=("material_id", "nunique"), value=("closing_stock_value", "sum"))
    rows = [{"name": str(r["aging_risk"]), "skus": int(r["skus"]), "value": float(r["value"])} for _, r in g.iterrows()]
    cols = [{"key": "name", "label": "Risk band", "kind": "text"}, {"key": "skus", "label": "SKUs", "kind": "num"},
            {"key": "value", "label": "Value", "kind": "inr"}]
    chart = {"type": "pie", "x": "name", "y": "skus", "title": "Stock risk mix"}
    return _result("Stock risk summary", rows, cols,
                   {"total_skus": int(df["material_id"].nunique()),
                    "reorder_now": int((df["replenishment_quantity"] > 0).sum())}, chart)


# ───────────────────────── FORECAST ─────────────────────────
def forecast(view: str = "demand", top_n: int = 10, plant: str = None):
    """Forward-looking view. view: demand | risk | fulfillment."""
    view = (view or "demand").lower().strip()
    if view == "fulfillment":
        try:
            f = _plant_filter(_load("kpi_fulfillment"), plant)
            g = f.groupby("month", as_index=False)["fulfillment_rate"].mean() if "month" in f.columns else None
            if g is not None:
                rows = [{"name": str(r["month"]), "fulfillment_pct": round(float(r["fulfillment_rate"]) * 100, 1)} for _, r in g.iterrows()]
                cols = [{"key": "name", "label": "Month", "kind": "text"}, {"key": "fulfillment_pct", "label": "Fulfillment %", "kind": "pct"}]
                return _result("Fulfillment rate trend", rows, cols, {},
                               {"type": "line", "x": "name", "y": "fulfillment_pct", "title": "Fulfillment rate"})
        except Exception:
            pass

    # demand / risk radar
    r = _plant_filter(da.load("kpi_stock_radar"), plant).copy()
    g = r.groupby("radar_status", as_index=False).agg(skus=("material_id", "nunique"),
                                                      demand=("demand_forecast", "sum"))
    rows = [{"name": str(x["radar_status"]), "skus": int(x["skus"]), "forecast_demand": float(x["demand"])} for _, x in g.iterrows()]
    cols = [{"key": "name", "label": "Status", "kind": "text"}, {"key": "skus", "label": "SKUs", "kind": "num"},
            {"key": "forecast_demand", "label": "Forecast demand", "kind": "num"}]
    chart = {"type": "pie", "x": "name", "y": "skus", "title": "Forecast risk radar"}
    return _result("Forecast stock radar", rows, cols, {"total_skus": int(r["material_id"].nunique())}, chart)


# ───────────────────────── HEADLINE FACTS (fast, no dimension) ─────────────────────────
def overview():
    """One-shot portfolio headline numbers — for 'how are we doing' / summary questions."""
    stats = {}
    try:
        s = _load("sales_totals")
        rev = float(s["revenue"].sum()); cost = float(s["cost"].sum())
        stats["revenue"] = rev; stats["margin"] = rev - cost
        stats["margin_pct"] = (rev - cost) / rev * 100 if rev else 0
    except Exception:
        pass
    try:
        stats["stock_value"] = float(_load("kpi_stock_value")["stock_value_cost"].sum())
    except Exception:
        pass
    try:
        stats["purchase_value"] = float(_load("kpi_purchase_value")["purchase_value"].sum())
    except Exception:
        pass
    try:
        ex = expiry()
        stats["expiry_within_180d"] = ex["stats"].get("actionable_within_180d", 0)
    except Exception:
        pass
    rows = [{"name": k.replace("_", " ").title(), "value": v} for k, v in stats.items()]
    cols = [{"key": "name", "label": "Metric", "kind": "text"}, {"key": "value", "label": "Value", "kind": "num"}]
    return _result("Portfolio overview", rows, cols, stats, None,
                   "Headline figures across revenue, inventory, procurement and expiry.")


# ───────────────────────── TOOL REGISTRY ─────────────────────────
# name → (callable, json-schema params) for the LLM function-calling layer.
TOOLS = {
    "revenue": revenue,
    "procurement": procurement,
    "inventory": inventory,
    "expiry": expiry,
    "stock_risk": stock_risk,
    "forecast": forecast,
    "overview": overview,
}


def run_tool(name: str, args: dict) -> dict:
    fn = TOOLS.get(name)
    if not fn:
        return _result("Unknown query", [], [], note=f"No analytics function '{name}'.")
    try:
        return fn(**(args or {}))
    except TypeError:
        # drop unexpected kwargs
        import inspect
        ok = {k: v for k, v in (args or {}).items() if k in inspect.signature(fn).parameters}
        return fn(**ok)
    except Exception as e:
        return _result("Query failed", [], [], note=f"Could not compute: {e}")
