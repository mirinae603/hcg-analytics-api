"""Original-frontend KPI contract served from the parquet layer.

Reproduces the exact endpoint paths + response shapes the original UI components
consume: chart endpoints return Title-Case keyed records, table endpoints return
{data,total} with camelCase keys (matching the MaterialReactTable accessorKeys).
Plant param may be a plant code or a region name (name => all plants).
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from typing import Optional
from functools import lru_cache
import os
import numpy as np
import pandas as pd

from app.core import data_access as da

# Portfolio-overview results are pure functions of the STATIC snapshot parquet, so the
# output for a given plant never changes until the data is refreshed. We memoize the
# whole computation per normalized plant — the first call warms it (and pulls the heavy
# fact_grn/fact_po parquet), every later call is instant. refresh_cache() clears these.
_RESULT_CACHES: list = []


def _cache(fn):
    cached = lru_cache(maxsize=64)(fn)
    _RESULT_CACHES.append(cached)
    return cached


def clear_result_caches() -> None:
    for c in _RESULT_CACHES:
        c.cache_clear()


def warmup() -> None:
    """Precompute the default (All-Plants) portfolio overviews so the FIRST request
    after a cold start hits a warm cache instead of loading fact_grn/fact_po live.
    Names resolve at call time (the cached fns are defined below)."""
    for fn_name, args in (
        ("_procurement_overview_cached", (None,)),
        ("_procurement_savings_cached", (None, 12)),
        ("_consumption_overview_cached", (None,)),
        ("_forecasting_overview_cached", (None,)),
    ):
        try:
            globals()[fn_name](*args)
        except Exception:
            pass

_KPI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kpi")
_MN = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
       "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}

router = APIRouter()

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def _plant(p: Optional[str]) -> Optional[str]:
    return da.resolve_plant(p)


def _filt(df, plant, material, group):
    df = da.filter_plant(df, _plant(plant))
    if material and material != "All Items" and "material" in df.columns:
        df = df[df["material"].astype(str).isin([m.strip() for m in str(material).split(",")])]
    elif group and "material_group" in df.columns:
        df = df[df["material_group"].astype(str) == str(group)]
    return df


def _monthly(df, measure, label, keep_group=True):
    """Aggregate to monthly series with Title-Case keys the charts expect."""
    gcols = ["year", "month"] + (["material_group"] if keep_group and "material_group" in df.columns else [])
    g = df.groupby(gcols, as_index=False, observed=True)[measure].sum()
    g["_mk"] = g["month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    g = g.sort_values(["year", "_mk"]).drop(columns="_mk")
    out = []
    for _, r in g.iterrows():
        row = {"Year": int(r["year"]) if pd.notna(r["year"]) else None,
               "Month": r["month"], "Period": r["month"], label: float(r[measure])}
        if keep_group and "material_group" in g.columns:
            row["Material Group"] = r.get("material_group")
        out.append(row)
    return out


def _table(table, plant, params, colmap, rename):
    cols = list(rename.keys())
    return da.paginate(table, _plant(plant), params,
                       col_map={v: k for k, v in {**colmap}.items()}, columns=cols, rename=rename)


# ---------------- STOCK CHANGE (KPI_1) ----------------
@router.get("/kpi/stock-change")
def stock_change(Plant: str = Query(None), Material: str = Query(None),
                 MaterialGroup: str = Query(None), Frequency: str = Query("Monthly")):
    df = _filt(da.load("kpi_stock_change"), Plant, Material, MaterialGroup)
    return _monthly(df, "stock_change", "Stock Change")


@router.get("/kpi/stock-change-table")
def stock_change_table(request: Request, Plant: str = Query(None)):
    return da.paginate("kpi_stock_change", _plant(Plant), dict(request.query_params),
        col_map={"year": "year", "period": "month", "materialId": "material",
                 "materialName": "material_desc", "materialGroup": "material_group", "stockChange": "stock_change"},
        columns=["year", "month", "material", "material_desc", "material_group", "stock_change"],
        rename={"year": "year", "month": "period", "material": "materialId",
                "material_desc": "materialName", "material_group": "materialGroup", "stock_change": "stockChange"})


# ---------------- INVENTORY VALUATION (KPI_2) — snapshot proxy ----------------
@router.get("/kpi/inventory-valuation")
def inventory_valuation(Plant: str = Query(None), Material: str = Query(None), MaterialGroup: str = Query(None)):
    df = _filt(da.load("kpi_stock_value"), Plant, Material, MaterialGroup)
    g = df.groupby("material_group", as_index=False)["stock_value_cost"].sum()
    return [{"Year": 2026, "Month": "May", "Period": "May", "Material Group": r["material_group"],
             "Inventory Valuation": float(r["stock_value_cost"])} for _, r in g.iterrows()]


@router.get("/kpi/inventory-valuation/insights")
def valuation_insights(Plant: str = Query(None)):
    inv = da.filter_plant(da.load("fact_inventory"), _plant(Plant)).copy()
    cost = float(inv["total_cost"].sum())
    mrp = float(inv["total_mrp"].sum())
    markup = mrp - cost
    markup_pct = (markup / cost * 100) if cost else 0.0
    skus = int(inv["material"].nunique())
    qty = float(inv["qty"].sum())

    # capital age profile — book value still on shelf by how long it's been held
    inv["aging_days"] = pd.to_numeric(inv["aging_days"], errors="coerce").fillna(0)
    labels = ["≤30d", "31–90d", "91–180d", "181–365d", "365d+"]
    bins = [-1, 30, 90, 180, 365, 1e12]
    inv["ageb"] = pd.cut(inv["aging_days"], bins=bins, labels=labels)
    ap = inv.groupby("ageb", observed=True).agg(
        cost=("total_cost", "sum"), mrp=("total_mrp", "sum"),
        qty=("qty", "sum"), skus=("material", "nunique")).reindex(labels).fillna(0).reset_index()
    age = [{"label": str(r["ageb"]), "cost": float(r["cost"]), "mrp": float(r["mrp"]),
            "qty": float(r["qty"]), "skus": int(r["skus"])} for _, r in ap.iterrows()]

    # where the capital sits — value by category (cost + retail), with markup
    g = inv.groupby("material_group", observed=True).agg(
        cost=("total_cost", "sum"), mrp=("total_mrp", "sum"), skus=("material", "nunique")).reset_index()
    g["markup_pct"] = np.where(g["cost"] > 0, (g["mrp"] - g["cost"]) / g["cost"] * 100, 0.0)
    cats = [{"name": r["material_group"], "cost": float(r["cost"]), "mrp": float(r["mrp"]),
             "markup": float(r["mrp"] - r["cost"]), "markup_pct": float(r["markup_pct"]), "skus": int(r["skus"])}
            for _, r in g.sort_values("cost", ascending=False).head(12).iterrows()]

    return {
        "totals": {"cost": cost, "mrp": mrp, "markup": markup, "markup_pct": markup_pct, "skus": skus, "qty": qty},
        "age": age, "categories": cats,
    }


@router.get("/kpi/inventory-valuation-table")
def inventory_valuation_table(request: Request, Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_stock_value"), _plant(Plant)).copy()
    df["year"] = 2026; df["period"] = "May"
    return _paginate_df(df, request, ["year", "period", "material", "material_desc", "material_group", "stock_value_cost"],
        {"year": "year", "period": "period", "material": "materialId", "material_desc": "materialName",
         "material_group": "materialGroup", "stock_value_cost": "inventoryValuation"})


# ---------------- INVENTORY TURNOVER — velocity insights (A4) ----------------
# ITR = annualized COGS / inventory value. COGS = 6-month consumption cost, so we
# annualize (×2). Single snapshot ⇒ "average inventory" is the snapshot point (proxy).
@router.get("/kpi/inventory-turnover-ratio/insights")
def itr_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_health_score"), _plant(Plant)).copy()
    ANN = 2.0  # 6 months -> annual
    cogs6 = float(df["consumption_cost"].sum())
    inv = float(df["closing_stock_value"].sum())
    port_itr = (cogs6 * ANN) / inv if inv else 0.0
    tv = df["turnover_annualized"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    movers = df[df["consumption_cost"] > 0]
    movers_itr = float(movers["turnover_annualized"].replace([np.inf, -np.inf], np.nan).clip(upper=60).mean()) if len(movers) else 0.0

    g = df.groupby("material_group", observed=True).agg(
        cogs=("consumption_cost", "sum"), inv=("closing_stock_value", "sum"), skus=("material", "nunique")).reset_index()
    g["itr"] = np.where(g["inv"] > 0, g["cogs"] * ANN / g["inv"], 0.0)
    cats = [{"name": r["material_group"], "itr": float(r["itr"]), "cogs": float(r["cogs"]),
             "inv": float(r["inv"]), "skus": int(r["skus"])} for _, r in g.sort_values("inv", ascending=False).head(12).iterrows()]

    def band(lo, hi):
        seg = df[(tv > lo) & (tv <= hi)]
        return {"count": int(len(seg)), "value": float(seg["closing_stock_value"].sum())}
    dead = df[tv <= 0]
    bands = [
        {"key": "dead", "label": "Dead (0×)", "count": int(len(dead)), "value": float(dead["closing_stock_value"].sum())},
        {"key": "slow", "label": "Slow (≤1×)", **band(0, 1)},
        {"key": "moderate", "label": "Moderate (1–4×)", **band(1, 4)},
        {"key": "fast", "label": "Fast (>4×)", **band(4, 1e12)},
    ]
    # monthly consumption-cost flow (the turnover engine) — smooth trend line
    fc = da.filter_plant(da.load("fact_consumption"), _plant(Plant))
    tl = fc.groupby(["year", "month_num", "month"], observed=True)["amount_lc"].sum().reset_index().sort_values(["year", "month_num"])
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "value": float(r["amount_lc"])} for _, r in tl.iterrows()]

    return {
        "totals": {"portfolio_itr": port_itr, "cogs_6mo": cogs6, "cogs_annual": cogs6 * ANN, "inventory": inv,
                   "months_on_hand": (12.0 / port_itr) if port_itr else 0.0, "total_skus": int(len(df)),
                   "movers": int(len(movers)), "movers_itr": movers_itr},
        "categories": cats, "bands": bands, "timeline": timeline,
    }


# ---------------- INVENTORY TURNOVER (KPI_3) — snapshot proxy ----------------
@router.get("/kpi/inventory-turnover-ratio")
def itr(Plant: str = Query(None), Material: str = Query(None), MaterialGroup: str = Query(None)):
    df = _filt(da.load("kpi_health_score"), Plant, Material, MaterialGroup)
    g = df.groupby("material_group", as_index=False)["turnover_annualized"].mean()
    return [{"Year": 2026, "Month": "May", "Period": "May", "Material Group": r["material_group"],
             "ITR": round(float(r["turnover_annualized"]), 2)} for _, r in g.iterrows()]


@router.get("/kpi/inventory-turnover-ratio-table")
def itr_table(request: Request, Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_health_score"), _plant(Plant)).copy()
    df["year"] = 2026; df["period"] = "May"
    return _paginate_df(df, request, ["year", "period", "material", "material_desc", "material_group", "consumption_cost", "closing_stock_value", "turnover_annualized"],
        {"year": "year", "period": "period", "material": "materialId", "material_desc": "materialName",
         "material_group": "materialGroup", "consumption_cost": "cogs", "closing_stock_value": "averageInventoryValue", "turnover_annualized": "inventoryTurnOverRatio"})


# ---------------- RETURN RATE (KPI_4) — no returns => zero proxy ----------------
@router.get("/kpi/return-rate")
def return_rate(Plant: str = Query(None), Material: str = Query(None), MaterialGroup: str = Query(None)):
    df = _filt(da.load("kpi_units_consumed"), Plant, Material, MaterialGroup)
    rows = _monthly(df, "total_units", "Return Rate (%)")
    for r in rows:
        r["Return Rate (%)"] = 0.0
    return rows


@router.get("/kpi/return-rate-table")
def return_rate_table(request: Request, Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_units_consumed"), _plant(Plant)).copy()
    df["returned"] = 0; df["rate"] = 0.0
    return _paginate_df(df, request, ["year", "month", "material", "material_desc", "material_group", "total_units", "returned", "rate"],
        {"year": "year", "month": "period", "material": "materialId", "material_desc": "materialName",
         "material_group": "materialGroup", "total_units": "soldUnits", "returned": "returnedUnits", "rate": "returnRate"})


# ---------------- DAYS OF INVENTORY ON HAND — coverage insights (A3) ----------------
# DOH = days of forward cover = stock_qty / avg_daily_consumption. A diverging story:
# stockout-risk (too little) ←→ overstock/idle capital (too much), plus a large
# non-moving segment (no demand => undefined cover). Value is joined from stock-value.
DOH_BANDS = [
    ("critical", "< 15 days", 0, 15),
    ("low", "15–30 days", 15, 30),
    ("healthy", "30–90 days", 30, 90),
    ("ample", "90–180 days", 90, 180),
    ("excess", "180–365 days", 180, 365),
    ("overstock", "365+ days", 365, float("inf")),
]


@router.get("/kpi/days-on-hand/insights")
def doh_insights(Plant: str = Query(None)):
    doh = da.filter_plant(da.load("kpi_doh"), _plant(Plant)).copy()
    sv = da.filter_plant(da.load("kpi_stock_value"), _plant(Plant))[["plant", "material", "stock_value_cost"]]
    doh = doh.merge(sv, on=["plant", "material"], how="left")
    doh["stock_value_cost"] = doh["stock_value_cost"].fillna(0.0)
    moving = doh[doh["doh_days"].notna()].copy()
    nonmoving = doh[doh["doh_days"].isna()]

    bands = []
    for key, label, lo, hi in DOH_BANDS:
        seg = moving[(moving["doh_days"] >= lo) & (moving["doh_days"] < hi)]
        bands.append({"key": key, "label": label, "count": int(len(seg)),
                      "value": float(seg["stock_value_cost"].sum()), "qty": float(seg["stock_qty"].sum())})
    bands.append({"key": "nonmoving", "label": "No movement", "count": int(len(nonmoving)),
                  "value": float(nonmoving["stock_value_cost"].sum()), "qty": float(nonmoving["stock_qty"].sum())})

    cat = doh.groupby("material_group", as_index=False, observed=True).agg(
        stock_qty=("stock_qty", "sum"), daily=("avg_daily_consumption", "sum"),
        value=("stock_value_cost", "sum"), skus=("material", "nunique"))
    cat["doh"] = np.where(cat["daily"] > 0, cat["stock_qty"] / cat["daily"], np.nan)
    cat = cat.sort_values("value", ascending=False).head(12)
    categories = [{"name": r["material_group"], "doh": (None if pd.isna(r["doh"]) else float(r["doh"])),
                   "value": float(r["value"]), "qty": float(r["stock_qty"]),
                   "daily": float(r["daily"]), "skus": int(r["skus"])} for _, r in cat.iterrows()]

    risk = moving[moving["avg_daily_consumption"] > 0].sort_values("doh_days").head(12)
    reorder = [{"name": r["material_desc"], "material": r["material"], "doh": float(r["doh_days"]),
                "value": float(r["stock_value_cost"]), "qty": float(r["stock_qty"]),
                "daily": float(r["avg_daily_consumption"])} for _, r in risk.iterrows()]

    over = moving[moving["doh_days"] > 365].sort_values("stock_value_cost", ascending=False).head(8)
    overstock = [{"name": r["material_desc"], "material": r["material"], "doh": float(r["doh_days"]),
                  "value": float(r["stock_value_cost"]), "qty": float(r["stock_qty"])} for _, r in over.iterrows()]

    bv = {b["key"]: b for b in bands}
    return {
        "totals": {
            "total_skus": int(len(doh)), "moving_skus": int(len(moving)), "nonmoving_skus": int(len(nonmoving)),
            "median_doh": (float(moving["doh_days"].median()) if len(moving) else 0.0),
            "mean_doh": (float(moving["doh_days"].mean()) if len(moving) else 0.0),
            "total_value": float(doh["stock_value_cost"].sum()),
            "risk_value": bv["critical"]["value"] + bv["low"]["value"],
            "risk_count": bv["critical"]["count"] + bv["low"]["count"],
            "healthy_value": bv["healthy"]["value"],
            "overstock_value": bv["overstock"]["value"],
            "overstock_count": bv["overstock"]["count"],
            "nonmoving_value": bv["nonmoving"]["value"],
        },
        "bands": bands, "categories": categories, "reorder": reorder, "overstock": overstock,
    }


# ---------------- INVENTORY HEALTH SCORE — composite vitals (A8) ----------------
# Health score (0–100) blends aging, turnover and movement into a tier classification.
# Generic sum-based endpoints can't give per-tier / per-category AVERAGES, so compute here.
HS_TIERS = ["Healthy", "Watch", "At Risk"]


@router.get("/kpi/inventory-health-score/insights")
def health_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_health_score"), _plant(Plant)).copy()
    df["moving"] = (df["consumption_cost"] > 0).astype(int)
    n = len(df)
    total_value = float(df["closing_stock_value"].sum())

    tiers = []
    for t in HS_TIERS:
        seg = df[df["health_tier"] == t]
        tiers.append({
            "tier": t, "count": int(len(seg)), "value": float(seg["closing_stock_value"].sum()),
            "avg_score": float(seg["health_score"].mean()) if len(seg) else 0.0,
            "avg_aging": float(seg["aging_days"].mean()) if len(seg) else 0.0,
            "avg_turnover": float(seg["turnover_annualized"].median()) if len(seg) else 0.0,
            "moving_pct": float(seg["moving"].mean()) if len(seg) else 0.0,
        })

    bands = []
    for lo, hi, lab in [(0, 20, "0–20"), (20, 40, "20–40"), (40, 60, "40–60"), (60, 80, "60–80"), (80, 100.01, "80–100")]:
        seg = df[(df["health_score"] >= lo) & (df["health_score"] < hi)]
        bands.append({"label": lab, "count": int(len(seg)), "value": float(seg["closing_stock_value"].sum())})

    g = df.groupby("material_group", observed=True).agg(
        avg_score=("health_score", "mean"), value=("closing_stock_value", "sum"),
        skus=("material", "nunique"), moving=("moving", "mean"), avg_aging=("aging_days", "mean")).reset_index()
    mix = df.pivot_table(index="material_group", columns="health_tier", values="material",
                         aggfunc="count", fill_value=0, observed=True)
    cats = []
    for _, r in g.sort_values("value", ascending=False).head(12).iterrows():
        mg = r["material_group"]; row = mix.loc[mg] if mg in mix.index else None
        cats.append({"name": mg, "avg_score": float(r["avg_score"]), "value": float(r["value"]),
                     "skus": int(r["skus"]), "moving": float(r["moving"]), "avg_aging": float(r["avg_aging"]),
                     "mix": {"healthy": int(row.get("Healthy", 0)) if row is not None else 0,
                             "watch": int(row.get("Watch", 0)) if row is not None else 0,
                             "atrisk": int(row.get("At Risk", 0)) if row is not None else 0}})

    return {
        "totals": {"avg_score": float(df["health_score"].mean()) if n else 0.0,
                   "median_score": float(df["health_score"].median()) if n else 0.0,
                   "total_skus": n, "total_value": total_value,
                   "moving_pct": float(df["moving"].mean()) if n else 0.0,
                   "fresh_pct": float((df["aging_days"] <= 90).mean()) if n else 0.0},
        "tiers": tiers, "bands": bands, "categories": cats,
    }


# ---------------- STOCK LEVEL CHANGE OVER TIME — monthly flow (A6) ----------------
@router.get("/kpi/stock-change/insights")
def stockchange_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_stock_change"), _plant(Plant)).copy()
    m = df.groupby(["year", "month"], observed=True).agg(
        inflow=("inflow", "sum"), outflow=("outflow", "sum"), net=("stock_change", "sum")).reset_index()
    m["_k"] = m["month"].map({mm: i for i, mm in enumerate(MONTH_ORDER)})
    m = m.sort_values(["year", "_k"])
    months = [{"label": f"{str(r['month'])[:3]} {str(int(r['year']))[2:]}", "month": r["month"],
               "inflow": float(r["inflow"]), "outflow": float(r["outflow"]), "net": float(r["net"])} for _, r in m.iterrows()]
    g = df.groupby(["material", "material_desc", "material_group"], observed=True).agg(
        inflow=("inflow", "sum"), outflow=("outflow", "sum"), net=("stock_change", "sum")).reset_index()

    def rows(x):
        return [{"name": r["material_desc"], "material": r["material"], "inflow": float(r["inflow"]),
                 "outflow": float(r["outflow"]), "net": float(r["net"])} for _, r in x.iterrows()]
    return {"totals": {"inflow": float(df["inflow"].sum()), "outflow": float(df["outflow"].sum()),
                       "net": float(df["stock_change"].sum()), "skus": int(df["material"].nunique()),
                       "months": int(df["month"].nunique())},
            "monthly": months, "risers": rows(g.sort_values("net", ascending=False).head(8)),
            "fallers": rows(g.sort_values("net").head(8))}


# ---------------- NON-MOVING INVENTORY — blocked capital (A9) ----------------
@router.get("/kpi/non-moving-inventory/insights")
def nonmoving_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_non_moving"), _plant(Plant)).copy()

    def band(a):
        if a < 90: return "≤90d"
        if a < 180: return "91–180d"
        if a < 365: return "181–365d"
        return "365d+"
    df["aband"] = df["aging_days"].apply(band)
    reasons = [{"reason": r, "count": int((df["reason"] == r).sum()),
                "value": float(df.loc[df["reason"] == r, "closing_stock_value"].sum())}
               for r in df["reason"].unique()]
    cat = df.groupby("material_group", observed=True).agg(
        value=("closing_stock_value", "sum"), skus=("material", "nunique"), qty=("closing_stock_quantity", "sum")
    ).reset_index().sort_values("value", ascending=False).head(12)
    cats = [{"name": r["material_group"], "value": float(r["value"]), "skus": int(r["skus"]), "qty": float(r["qty"])} for _, r in cat.iterrows()]
    order = ["≤90d", "91–180d", "181–365d", "365d+"]
    ag = df.groupby("aband", observed=True).agg(value=("closing_stock_value", "sum"), count=("material", "count"))
    aging = [{"label": l, "value": float(ag.loc[l, "value"]) if l in ag.index else 0.0,
              "count": int(ag.loc[l, "count"]) if l in ag.index else 0} for l in order]
    return {"totals": {"blocked_value": float(df["closing_stock_value"].sum()), "blocked_skus": int(len(df)),
                       "qty": float(df["closing_stock_quantity"].sum())},
            "reasons": reasons, "categories": cats, "aging": aging}


# ---------------- INVENTORY RISK CLASSIFICATION — risk matrix (A10) ----------------
@router.get("/kpi/inventory-risk/insights")
def risk_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_risk_classification"), _plant(Plant)).copy()
    tiers = [{"level": L, "count": int((df["risk_level"] == L).sum()),
              "value": float(df.loc[df["risk_level"] == L, "closing_stock_value"].sum())}
             for L in ["High", "Medium", "Low"]]

    def ab(a):
        if a < 30: return "0–30"
        if a < 90: return "31–90"
        if a < 180: return "91–180"
        if a < 365: return "181–365"
        return "365+"

    def eb(d):
        if pd.isna(d): return "No date"
        if d < 0: return "Expired"
        if d <= 90: return "≤90d"
        if d <= 365: return "91–365d"
        return "365d+"
    df["ab"] = df["aging_days"].apply(ab); df["eb"] = df["days_to_expiry"].apply(eb)
    arows = ["0–30", "31–90", "91–180", "181–365", "365+"]; ecols = ["Expired", "≤90d", "91–365d", "365d+", "No date"]
    mat = df.pivot_table(index="ab", columns="eb", values="closing_stock_value", aggfunc="sum", fill_value=0, observed=True).reindex(index=arows, columns=ecols, fill_value=0)
    matrix = [[float(mat.loc[a, e]) for e in ecols] for a in arows]
    hi = df[df["risk_level"] == "High"]
    factors = [{"label": "Expiring ≤90 days", "count": int((hi["days_to_expiry"] <= 90).sum())},
               {"label": "Aged > 180 days", "count": int((hi["aging_days"] > 180).sum())},
               {"label": "No consumption", "count": int((~hi["consumed"].astype(bool)).sum())}]
    cat = hi.groupby("material_group", observed=True).agg(value=("closing_stock_value", "sum"), skus=("material", "nunique")).reset_index().sort_values("value", ascending=False).head(10)
    cats = [{"name": r["material_group"], "value": float(r["value"]), "skus": int(r["skus"])} for _, r in cat.iterrows()]
    return {"totals": {"total_value": float(df["closing_stock_value"].sum()), "total_skus": int(len(df))},
            "tiers": tiers, "arows": arows, "ecols": ecols, "matrix": matrix, "factors": factors, "categories": cats}


# ---------------- NEAR-EXPIRY — full expiry ladder (client #3: 5-slab breakup) ----------------
# The kpi_near_expiry parquet is horizon-limited to <=180 days, so the client's
# 181-365 / 365+ slabs are computed straight off fact_inventory vs the snapshot date.
_EXP_SLAB_ORDER = ["Expired", "0-30d", "31-90d", "91-180d", "181-365d", "365d+"]


def _inv_expiry(pl):
    """fact_inventory (plant-filtered, positive qty, known expiry) + days-to-expiry & slab."""
    df = da.filter_plant(da.load("fact_inventory"), pl).copy()
    df = df.dropna(subset=["expiry_date"])
    df = df[df["qty"] > 0]
    snap = pd.to_datetime(df["snapshot_date"], errors="coerce").max()
    if pd.isna(snap):
        snap = pd.Timestamp("2026-05-31")
    df["dte"] = (pd.to_datetime(df["expiry_date"], errors="coerce") - snap).dt.days
    df = df.dropna(subset=["dte"])
    bins = [-10**12, -1, 30, 90, 180, 365, 10**12]
    df["slab"] = pd.cut(df["dte"], bins=bins, labels=_EXP_SLAB_ORDER, right=True)
    return df, snap


def _expiry_ladder(df):
    out = []
    for s in _EXP_SLAB_ORDER:
        sub = df[df["slab"] == s]
        out.append({"slab": s, "lines": int(len(sub)), "items": int(sub["material"].nunique()) if len(sub) else 0,
                    "qty": float(sub["qty"].sum()), "value": float(sub["total_cost"].sum()),
                    "mrp": float(sub["total_mrp"].sum())})
    return out


# ---------------- NEAR-EXPIRY — expiry timeline (E1) ----------------
@router.get("/kpi/near-expiry/insights")
def nearexp_insights(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_near_expiry"), _plant(Plant)).copy()
    buckets = [{"bucket": b, "count": int((df["expiry_bucket"] == b).sum()),
                "value": float(df.loc[df["expiry_bucket"] == b, "total_cost"].sum()),
                "mrp": float(df.loc[df["expiry_bucket"] == b, "total_mrp"].sum()),
                "qty": float(df.loc[df["expiry_bucket"] == b, "qty"].sum())}
               for b in ["Expired", "0-30d", "31-90d", "91-180d"] if (df["expiry_bucket"] == b).any()]
    # forward expiry calendar — upcoming months only (exclude already-expired/odd past dates)
    d = df[df["days_to_expiry"] >= 0].copy(); d["expiry_date"] = pd.to_datetime(d["expiry_date"], errors="coerce")
    d = d.dropna(subset=["expiry_date"]); d["period"] = d["expiry_date"].dt.to_period("M")
    tl = d.groupby("period", observed=True).agg(value=("total_cost", "sum"), count=("material", "count")).reset_index().sort_values("period")
    timeline = [{"label": p.strftime("%b %y"), "value": float(v), "count": int(c)} for p, v, c in zip(tl["period"], tl["value"], tl["count"])][:10]
    cat = df.groupby("material_group", observed=True).agg(value=("total_cost", "sum"), skus=("material", "nunique")).reset_index().sort_values("value", ascending=False).head(10)
    cats = [{"name": r["material_group"], "value": float(r["value"]), "skus": int(r["skus"])} for _, r in cat.iterrows()]
    # Full expiry ladder (all 6 slabs incl. 181-365 / 365+) from fact_inventory.
    try:
        invdf, snap = _inv_expiry(_plant(Plant))
        ladder = _expiry_ladder(invdf)
        ladder_asof = snap.strftime("%d %b %Y") if snap is not None else None
    except Exception:
        ladder, ladder_asof = [], None
    return {"totals": {"exposure": float(df["total_cost"].sum()), "skus": int(len(df)),
                       "expired_value": float(df.loc[df["expiry_bucket"] == "Expired", "total_cost"].sum()),
                       "urgent_value": float(df.loc[df["expiry_bucket"] == "0-30d", "total_cost"].sum()),
                       "mrp_exposure": float(df["total_mrp"].sum())},
            "buckets": buckets, "timeline": timeline, "categories": cats,
            "ladder": ladder, "ladder_asof": ladder_asof}


@router.get("/kpi/near-expiry/items")
def nearexp_items(slab: str = Query(None), Plant: str = Query(None),
                  q: str = Query(None), limit: int = Query(500)):
    """Drill: item lines for an expiry slab (client #3 — clickable slab → item list)."""
    invdf, snap = _inv_expiry(_plant(Plant))
    if slab:
        invdf = invdf[invdf["slab"].astype(str) == slab]
    if q:
        ql = str(q).strip().lower()
        invdf = invdf[invdf["material_desc"].astype(str).str.lower().str.contains(ql, na=False)
                      | invdf["material"].astype(str).str.lower().str.contains(ql, na=False)]
    total = int(len(invdf))
    invdf = invdf.sort_values("total_cost", ascending=False).head(int(limit))
    items = [{"material": r["material"], "desc": r["material_desc"],
              "batch": (None if pd.isna(r.get("batch")) else str(r.get("batch"))),
              "plant": r["plant"], "qty": float(r["qty"]), "value": float(r["total_cost"]),
              "mrp": float(r["total_mrp"]), "days": int(r["dte"]),
              "expiry": (pd.to_datetime(r["expiry_date"]).strftime("%d %b %Y") if pd.notna(r["expiry_date"]) else None),
              "slab": str(r["slab"])} for _, r in invdf.iterrows()]
    return {"slab": slab, "count": total, "returned": len(items), "items": items,
            "asof": (snap.strftime("%d %b %Y") if snap is not None else None)}


# ---------------- INVENTORY AGING (KPI_5 / Chart_KPI_5) ----------------
@router.get("/kpi/inventory-aging")
def inventory_aging(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_inventory_aging"), _plant(Plant)).copy()
    df = df.sort_values("closing_stock_value", ascending=False).head(50).reset_index(drop=True)
    out = []
    for i, r in df.iterrows():
        out.append({"Material Name": r["material_desc"], "Material ID": r["material"],
                    "Age Since Last Sale (days)": _num(r["age_since_last_sale_days"]),
                    "Age Since Last Purchase (days)": _num(r["age_since_last_purchase_days"]),
                    "Aging": _num(r["aging_days"]), "aging_category": r["aging_category"],
                    "closing_stock_value": float(r["closing_stock_value"]),
                    "closing_stock_quantity": float(r["closing_stock_quantity"]), "rank": int(i) + 1})
    return out


@router.get("/kpi/inventory-aging-table")
def inventory_aging_table(request: Request, Plant: str = Query(None)):
    return _paginate_df(da.filter_plant(da.load("kpi_inventory_aging"), _plant(Plant)), request,
        ["material", "material_desc", "age_since_last_sale_days", "age_since_last_purchase_days", "aging_days"],
        {"material": "materialId", "material_desc": "materialName",
         "age_since_last_sale_days": "ageSinceLastSale", "age_since_last_purchase_days": "ageSinceLastPurchase", "aging_days": "aging"})


# ---------------- PROCUREMENT OVERVIEW (portfolio dashboard) ----------------
@router.get("/portfolio/procurement/overview")
def procurement_overview(Plant: str = Query(None)):
    return _procurement_overview_cached(_plant(Plant))


@_cache
def _procurement_overview_cached(pl):
    pv = da.filter_plant(da.load("kpi_purchase_value"), pl)
    vv = da.filter_plant(da.load("kpi_vendor_volume"), pl)
    ct = da.filter_plant(da.load("kpi_cycle_time"), pl)
    fr = da.filter_plant(da.load("kpi_fill_rate"), pl)
    loc = da.filter_plant(da.load("kpi_purchase_by_location"), pl)
    lt = da.load("kpi_vendor_lead_time")  # vendor lead time has no plant dimension

    spend = float(pv["purchase_value"].sum())
    po_lines = int(pv["po_lines"].sum())

    vg = vv.groupby("vendor_name", observed=True).agg(val=("vendor_value", "sum"), lines=("po_lines", "sum")).reset_index().sort_values("val", ascending=False)
    n_vendors = int(vg["vendor_name"].nunique())
    vg["share"] = np.where(spend > 0, vg["val"] / spend * 100, 0.0)
    top_vendors = [{"name": str(r["vendor_name"]), "value": float(r["val"]), "share": float(r["share"]), "lines": int(r["lines"])} for _, r in vg.head(8).iterrows()]
    top5_share = float(vg.head(5)["val"].sum() / spend * 100) if spend else 0.0

    gl = float(ct["gr_lines"].sum())
    avg_po_gr = float((ct["avg_po_to_gr_tat"] * ct["gr_lines"]).sum() / gl) if gl else 0.0
    avg_pr_gr = float((ct["avg_pr_to_gr_tat"] * ct["gr_lines"]).sum() / gl) if gl else 0.0
    llg = float(lt["gr_lines"].sum())
    med_lead = float((lt["median_lead_time_days"] * lt["gr_lines"]).sum() / llg) if llg else 0.0
    ordered = float(fr["ordered_qty"].sum()); openq = float(fr["open_qty"].sum())
    completion = (max(0.0, min(1.0, 1 - openq / ordered)) * 100) if ordered else 0.0

    m = pv.groupby(["year", "month"], observed=True)["purchase_value"].sum().reset_index()
    m["_k"] = m["month"].map({mm: i for i, mm in enumerate(MONTH_ORDER)})
    m = m.sort_values(["year", "_k"])
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "value": float(r["purchase_value"])} for _, r in m.iterrows()]
    last_mom = 0.0
    if len(timeline) >= 2:
        a = timeline[-2]["value"]; b = timeline[-1]["value"]; last_mom = ((b - a) / a * 100) if a else 0.0

    cg = pv.groupby("category", observed=True)["purchase_value"].sum().reset_index().sort_values("purchase_value", ascending=False)
    categories = [{"name": str(r["category"]), "value": float(r["purchase_value"]),
                   "uncat": str(r["category"]).strip().lower() in ("uncategorized", "", "nan", "none")} for _, r in cg.head(10).iterrows()]

    lg = loc.sort_values("purchase_value", ascending=False)
    locations = [{"plant": str(r["plant"]), "value": float(r["purchase_value"]), "vendors": int(r["vendor_count"]), "lines": int(r["po_lines"])} for _, r in lg.head(8).iterrows()]
    n_plants = int(loc["plant"].nunique())

    # MRP-based procurement margin proxy (client #12: "add Margin % to the card").
    # Only rows where BOTH MRP and cost are recorded (unit_mrp is sparse in GRN, so
    # summing over all rows would count cost without its matching MRP → false negative).
    try:
        grn = da.filter_plant(da.load("fact_grn"), pl)
        g2 = grn[(grn["unit_mrp"] > 0) & (grn["net_price"] > 0) & (grn["gr_qty"] > 0)]
        mrp_val = float((g2["gr_qty"] * g2["unit_mrp"]).sum())
        cost_val = float((g2["gr_qty"] * g2["net_price"]).sum())
    except Exception:
        mrp_val = cost_val = 0.0
    margin_val = mrp_val - cost_val
    margin_pct = (margin_val / mrp_val * 100) if mrp_val else 0.0

    cards = {
        "purchase-value": {"value": spend, "kind": "inr", "sub": "6-mo PO spend"},
        "monthly-purchase-value": {"value": spend / max(len(timeline), 1), "kind": "inr", "sub": "avg / month"},
        "procurement-variance": {"value": last_mom, "kind": "pct", "sub": "latest MoM"},
        "vendor-volume-contribution": {"value": top5_share, "kind": "pct", "sub": "top-5 vendors"},
        "purchase-by-location": {"value": float(n_plants), "kind": "num", "sub": "plants"},
        "procurement-cycle-time": {"value": avg_po_gr, "kind": "days", "sub": "PO→GR"},
        "vendor-lead-time": {"value": med_lead, "kind": "days", "sub": "median lead"},
        "fill-rate": {"value": completion, "kind": "pct", "sub": "order completion"},
    }

    # Open POs — undelivered order value by category (client #12: category + number + value).
    try:
        po = da.filter_plant(da.load("fact_po"), pl)
        op = po[po["open_qty"] > 0].copy()
        op["open_value"] = op["open_qty"] * op["net_price"]
        op["g"] = op["major_group"].apply(_clean_group)
        opby = op.groupby("g").agg(po_count=("po_no", "nunique"), open_value=("open_value", "sum")).reset_index().sort_values("open_value", ascending=False)
        open_po = {"total_value": float(op["open_value"].sum()), "total_pos": int(op["po_no"].nunique()), "total_lines": int(len(op)),
                   "categories": [{"category": r["g"], "pos": int(r["po_count"]), "value": float(r["open_value"])} for _, r in opby.head(7).iterrows()]}
    except Exception:
        open_po = {"total_value": 0.0, "total_pos": 0, "total_lines": 0, "categories": []}

    return {
        "totals": {"spend": spend, "vendors": n_vendors, "po_lines": po_lines, "avg_po_gr": avg_po_gr,
                   "avg_pr_gr": avg_pr_gr, "median_lead": med_lead, "completion": completion,
                   "top5_share": top5_share, "n_plants": n_plants,
                   "margin_pct": margin_pct, "margin_value": margin_val, "mrp_value": mrp_val},
        "timeline": timeline, "categories": categories, "vendors": top_vendors, "locations": locations, "cards": cards,
        "open_po": open_po,
    }


# ---------------- PROCUREMENT SAVING OPPORTUNITY (client #12: item-wise loss/saving) ----------------
@router.get("/portfolio/procurement/savings")
def procurement_savings(Plant: str = Query(None), limit: int = Query(12)):
    """Price-consolidation headroom: for each material bought >=4 times at a
    consistent unit (max/min <= 2.5x, so we don't compare mixed pack sizes), sum the
    spend ABOVE that item's own median achieved price. Conservative negotiation
    headroom — an honest 'you paid above your own median' figure, not a guaranteed saving."""
    return _procurement_savings_cached(_plant(Plant), limit)


@_cache
def _procurement_savings_cached(pl, limit):
    empty = {"totals": {"opportunity": 0.0, "items_flagged": 0, "spend_base": 0.0}, "items": []}
    try:
        g = da.filter_plant(da.load("fact_grn"), pl)
        d = g[(g["net_price"] > 0) & (g["gr_qty"] > 0)][["material", "material_desc", "net_price", "gr_qty", "major_group"]].copy()
        if d.empty:
            return empty
        st = d.groupby("material")["net_price"].agg(["min", "max", "median", "size"])
        st = st[st["size"] >= 4]
        st = st[(st["max"] / st["min"].replace(0, np.nan)) <= 2.5].dropna(subset=["max"])
        if st.empty:
            return empty
        d2 = d[d["material"].isin(st.index)].merge(st[["median"]], left_on="material", right_index=True)
        d2["over"] = (d2["net_price"] - d2["median"]).clip(lower=0) * d2["gr_qty"]
        opp = d2.groupby(["material", "material_desc"]).agg(
            over=("over", "sum"), lines=("net_price", "size"), med=("median", "first"),
            pmax=("net_price", "max"), qty=("gr_qty", "sum"), group=("major_group", "first")).reset_index()
        opp = opp[opp["over"] > 0].sort_values("over", ascending=False)
        items = [{"material": r["material"], "desc": r["material_desc"], "group": _clean_group(r["group"]),
                  "median": float(r["med"]), "max": float(r["pmax"]), "lines": int(r["lines"]),
                  "qty": float(r["qty"]), "over": float(r["over"]),
                  "spread_pct": (float((r["pmax"] - r["med"]) / r["med"] * 100) if r["med"] else 0.0)}
                 for _, r in opp.head(int(limit)).iterrows()]
        return {"totals": {"opportunity": float(opp["over"].sum()), "items_flagged": int(len(opp)),
                           "spend_base": float((d2["net_price"] * d2["gr_qty"]).sum())}, "items": items}
    except Exception:
        return empty


# ---------------- PURCHASE VALUE insights (B1) ----------------
@router.get("/kpi/purchase-value/insights")
def purchase_value_insights(Plant: str = Query(None)):
    pl = _plant(Plant)
    pv = da.filter_plant(da.load("kpi_purchase_value"), pl)
    vv = da.filter_plant(da.load("kpi_vendor_volume"), pl)
    loc = da.filter_plant(da.load("kpi_purchase_by_location"), pl)

    spend = float(pv["purchase_value"].sum()); lines = int(pv["po_lines"].sum()); qty = float(pv["purchase_qty"].sum())
    avg_po = (spend / lines) if lines else 0.0

    vg = vv.groupby("vendor_name", observed=True).agg(val=("vendor_value", "sum"), lines=("po_lines", "sum")).reset_index().sort_values("val", ascending=False)
    n_vendors = int(vg["vendor_name"].nunique())
    vg["share"] = np.where(spend > 0, vg["val"] / spend * 100, 0.0)
    vendors = [{"name": str(r["vendor_name"]), "value": float(r["val"]), "share": float(r["share"]), "lines": int(r["lines"])} for _, r in vg.head(8).iterrows()]

    m = pv.groupby(["year", "month"], observed=True)["purchase_value"].sum().reset_index()
    m["_k"] = m["month"].map({mm: i for i, mm in enumerate(MONTH_ORDER)})
    m = m.sort_values(["year", "_k"])
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "value": float(r["purchase_value"])} for _, r in m.iterrows()]

    cg = pv.groupby("category", observed=True).agg(value=("purchase_value", "sum"), lines=("po_lines", "sum")).reset_index().sort_values("value", ascending=False)
    categories = [{"name": str(r["category"]), "value": float(r["value"]), "lines": int(r["lines"]),
                   "uncat": str(r["category"]).strip().lower() in ("uncategorized", "", "nan", "none")} for _, r in cg.head(12).iterrows()]

    lg = loc.sort_values("purchase_value", ascending=False)
    plants = [{"plant": str(r["plant"]), "value": float(r["purchase_value"]), "vendors": int(r["vendor_count"]), "lines": int(r["po_lines"])} for _, r in lg.head(8).iterrows()]

    return {"totals": {"spend": spend, "lines": lines, "qty": qty, "avg_po": avg_po, "vendors": n_vendors},
            "timeline": timeline, "categories": categories, "vendors": vendors, "plants": plants}


_PROC_WINDOW = {(2025, "December"), (2026, "January"), (2026, "February"), (2026, "March"), (2026, "April"), (2026, "May")}


# ---------------- PROCUREMENT VARIANCE insights (B3) ----------------
@router.get("/kpi/procurement-variance/insights")
def variance_insights(Plant: str = Query(None)):
    pv = da.filter_plant(da.load("kpi_purchase_value"), _plant(Plant))
    m = pv.groupby(["year", "month"], observed=True)["purchase_value"].sum().reset_index()
    m["_k"] = m["month"].map({x: i for i, x in enumerate(MONTH_ORDER)})
    m = m.sort_values(["year", "_k"])
    rows = []; prev = None
    for _, r in m.iterrows():
        v = float(r["purchase_value"])
        first = prev is None
        delta = 0.0 if first else (v - prev)
        pct = 0.0 if (first or not prev) else ((v - prev) / prev * 100)
        rows.append({"label": str(r["month"])[:3], "month": str(r["month"]), "value": v, "delta": delta, "pct": pct, "first": first})
        prev = v
    deltas = [x for x in rows if not x["first"]]
    up = max(deltas, key=lambda x: x["pct"]) if deltas else None
    down = min(deltas, key=lambda x: x["pct"]) if deltas else None
    last = rows[-1] if rows else {}
    vol = float(np.std([x["pct"] for x in deltas])) if deltas else 0.0
    return {"totals": {"latest_pct": last.get("pct", 0.0), "latest_delta": last.get("delta", 0.0),
                       "avg_spend": float(m["purchase_value"].mean()) if len(m) else 0.0,
                       "up_month": up["month"] if up else "-", "up_pct": up["pct"] if up else 0.0,
                       "down_month": down["month"] if down else "-", "down_pct": down["pct"] if down else 0.0,
                       "volatility": vol}, "timeline": rows}


# ---------------- VENDOR VOLUME insights (B4) ----------------
@router.get("/kpi/vendor-volume-contribution/insights")
def vendor_volume_insights(Plant: str = Query(None)):
    vv = da.filter_plant(da.load("kpi_vendor_volume"), _plant(Plant))
    vg = vv.groupby("vendor_name", observed=True).agg(value=("vendor_value", "sum"), lines=("po_lines", "sum"), qty=("vendor_qty", "sum")).reset_index().sort_values("value", ascending=False)
    tot = float(vg["value"].sum()); n = int(len(vg))
    vg["share"] = np.where(tot > 0, vg["value"] / tot * 100, 0.0)
    vg["cum"] = vg["share"].cumsum()
    n80 = int((vg["cum"] <= 80).sum()) + 1
    hhi = float((vg["share"] ** 2).sum())
    vendors = [{"name": str(r["vendor_name"]), "value": float(r["value"]), "share": float(r["share"]), "cum": float(r["cum"]), "lines": int(r["lines"])} for _, r in vg.head(15).iterrows()]
    return {"totals": {"vendors": n, "total": tot, "top1": float(vg["share"].iloc[0]) if n else 0.0,
                       "top5": float(vg.head(5)["share"].sum()), "top10": float(vg.head(10)["share"].sum()),
                       "n80": n80, "hhi": hhi}, "vendors": vendors}


# ---------------- PURCHASE BY LOCATION insights (B7) ----------------
@router.get("/kpi/purchase-by-location/insights")
def location_insights(Plant: str = Query(None)):
    loc = da.filter_plant(da.load("kpi_purchase_by_location"), _plant(Plant)).copy()
    tot = float(loc["purchase_value"].sum()); n = int(loc["plant"].nunique())
    loc["share"] = np.where(tot > 0, loc["purchase_value"] / tot * 100, 0.0)
    bypv = loc.sort_values("purchase_value", ascending=False)
    plants = [{"plant": str(r["plant"]), "value": float(r["purchase_value"]), "share": float(r["share"]), "vendors": int(r["vendor_count"]), "lines": int(r["po_lines"])} for _, r in bypv.head(12).iterrows()]
    byv = loc.sort_values("vendor_count", ascending=False)
    diverse = [{"plant": str(r["plant"]), "vendors": int(r["vendor_count"]), "value": float(r["purchase_value"])} for _, r in byv.head(7).iterrows()]
    top = bypv.iloc[0] if len(bypv) else None
    return {"totals": {"plants": n, "total": tot, "avg": (tot / n) if n else 0.0,
                       "top_plant": str(top["plant"]) if top is not None else "-",
                       "top_value": float(top["purchase_value"]) if top is not None else 0.0},
            "plants": plants, "diverse": diverse}


# ---------------- PROCUREMENT CYCLE TIME insights (E2) ----------------
@router.get("/kpi/procurement-cycle-time/insights")
def cycle_insights(Plant: str = Query(None)):
    ct = da.filter_plant(da.load("kpi_cycle_time"), _plant(Plant)).copy()
    ct = ct[ct.apply(lambda r: (int(r["year"]), str(r["month"])) in _PROC_WINDOW, axis=1)]
    ct["po_w"] = ct["avg_po_to_gr_tat"] * ct["gr_lines"]; ct["pr_w"] = ct["avg_pr_to_gr_tat"] * ct["gr_lines"]
    cm = ct.groupby(["year", "month"], observed=True).agg(po_w=("po_w", "sum"), pr_w=("pr_w", "sum"), gl=("gr_lines", "sum")).reset_index()
    cm["_k"] = cm["month"].map({x: i for i, x in enumerate(MONTH_ORDER)}); cm = cm.sort_values(["year", "_k"])
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]),
                 "po": float(r["po_w"] / r["gl"]) if r["gl"] else 0.0, "pr": float(r["pr_w"] / r["gl"]) if r["gl"] else 0.0} for _, r in cm.iterrows()]
    gl = float(ct["gr_lines"].sum())
    avg_po = float(ct["po_w"].sum() / gl) if gl else 0.0
    avg_pr = float(ct["pr_w"].sum() / gl) if gl else 0.0
    pp = ct.groupby("plant", observed=True).agg(po_w=("po_w", "sum"), gl=("gr_lines", "sum")).reset_index()
    pp["po"] = np.where(pp["gl"] > 0, pp["po_w"] / pp["gl"], 0.0)
    pp_sig = pp[pp["gl"] >= 20]  # drop tiny-sample plants that skew the ranking
    pp_sig = (pp_sig if len(pp_sig) else pp).sort_values("po", ascending=False)
    plants = [{"plant": str(r["plant"]), "po": float(r["po"]), "lines": int(r["gl"])} for _, r in pp_sig.head(8).iterrows()]
    fast = min(timeline, key=lambda x: x["po"]) if timeline else None
    slow = max(timeline, key=lambda x: x["po"]) if timeline else None
    return {"totals": {"avg_po": avg_po, "avg_pr": avg_pr,
                       "fast_month": fast["month"] if fast else "-", "fast_po": fast["po"] if fast else 0.0,
                       "slow_month": slow["month"] if slow else "-", "slow_po": slow["po"] if slow else 0.0},
            "timeline": timeline, "plants": plants}


# ---------------- VENDOR LEAD TIME insights (E3) ----------------
@router.get("/kpi/vendor-lead-time/insights")
def lead_insights(Plant: str = Query(None)):
    lt = da.load("kpi_vendor_lead_time").copy()  # no plant dimension
    gl = float(lt["gr_lines"].sum())
    med = float((lt["median_lead_time_days"] * lt["gr_lines"]).sum() / gl) if gl else 0.0
    labs = ["≤2d", "3–5d", "6–10d", "11–20d", "20d+"]; bins = [-1, 2, 5, 10, 20, 1e12]
    lt["b"] = pd.cut(lt["median_lead_time_days"], bins=bins, labels=labs)
    d = lt.groupby("b", observed=True).agg(lines=("gr_lines", "sum"), vendors=("vendor_name", "nunique")).reindex(labs).fillna(0).reset_index()
    dist = [{"label": str(r["b"]), "lines": int(r["lines"]), "vendors": int(r["vendors"])} for _, r in d.iterrows()]
    under7 = float(100 * lt[lt["median_lead_time_days"] <= 7]["gr_lines"].sum() / gl) if gl else 0.0
    sig = lt[lt["gr_lines"] >= 20]
    mk = lambda df: [{"name": str(r["vendor_name"]), "days": float(r["median_lead_time_days"]), "lines": int(r["gr_lines"])} for _, r in df.iterrows()]
    slow = mk(sig.sort_values("median_lead_time_days", ascending=False).head(8))
    fast = mk(sig.sort_values("median_lead_time_days").head(8))
    return {"totals": {"median": med, "under7": under7, "vendors": int(lt["vendor_name"].nunique()),
                       "fastest": fast[0] if fast else None, "slowest": slow[0] if slow else None},
            "dist": dist, "slow": slow, "fast": fast}


# ---------------- FILL RATE insights (E4) ----------------
@router.get("/kpi/fill-rate/insights")
def fill_insights(Plant: str = Query(None)):
    fr = da.filter_plant(da.load("kpi_fill_rate"), _plant(Plant)).copy()
    ordered = float(fr["ordered_qty"].sum()); openq = float(fr["open_qty"].sum())
    overall = (max(0.0, min(1.0, 1 - openq / ordered)) * 100) if ordered else 0.0
    fr["comp"] = (1 - fr["open_qty"] / fr["ordered_qty"]).clip(0, 1) * 100
    worst_df = fr.sort_values("comp")
    worst = [{"plant": str(r["plant"]), "comp": float(r["comp"]), "ordered": float(r["ordered_qty"]), "open": float(max(r["open_qty"], 0.0))} for _, r in worst_df.head(12).iterrows()]
    best = [{"plant": str(r["plant"]), "comp": float(r["comp"])} for _, r in fr.sort_values("comp", ascending=False).head(8).iterrows()]
    # every plant with real order volume — powers the priority scatter
    frv = fr[fr["ordered_qty"] > 0]
    plants = [{"plant": str(r["plant"]), "comp": float(r["comp"]), "ordered": float(r["ordered_qty"]), "open": float(max(r["open_qty"], 0.0))}
              for _, r in frv.sort_values("ordered_qty", ascending=False).head(60).iterrows()]
    # distribution of plants across fill-rate bands (best → worst)
    edges = [(99.5, 200.0, "100%"), (95.0, 99.5, "95–99%"), (85.0, 95.0, "85–95%"), (70.0, 85.0, "70–85%"), (-1.0, 70.0, "<70%")]
    dist = [{"label": lab, "plants": int(((frv["comp"] >= lo) & (frv["comp"] < hi)).sum())} for lo, hi, lab in edges]
    perfect = int((frv["comp"] >= 99.5).sum())
    return {"totals": {"overall": overall, "plants": int(fr["plant"].nunique()), "open_qty": openq, "ordered_qty": ordered,
                       "perfect": perfect, "best_plant": best[0]["plant"] if best else "-", "worst_plant": worst[0]["plant"] if worst else "-"},
            "worst": worst, "best": best, "plants": plants, "dist": dist}


# ---------------- MONTHLY SKU PURCHASE insights (B2) ----------------
@router.get("/kpi/monthly-purchase-value/insights")
def monthly_purchase_insights(Plant: str = Query(None)):
    mp = da.filter_plant(da.load("kpi_monthly_purchase_value"), _plant(Plant)).copy()
    total = float(mp["monthly_purchase_value"].sum())
    tg = mp.groupby("material_group", observed=True)["monthly_purchase_value"].sum().sort_values(ascending=False)
    top_groups = list(tg.head(4).index)
    # chronological order across the window (Dec-2025 → May-2026), not calendar-month index
    months = mp.groupby(["year", "month"], observed=True)["monthly_purchase_value"].sum().reset_index()
    months["_k"] = months["month"].map({x: i for i, x in enumerate(MONTH_ORDER)})
    months = months.sort_values(["year", "_k"])
    order = list(months["month"])
    total_series = [float(v) for v in months["monthly_purchase_value"]]
    series = []
    for g in top_groups:
        sub = mp[mp["material_group"] == g].groupby("month", observed=True)["monthly_purchase_value"].sum()
        series.append({"name": str(g), "values": [float(sub.get(m, 0)) for m in order]})
    sku = mp.groupby(["material", "material_desc", "material_group"], observed=True)["monthly_purchase_value"].sum().sort_values(ascending=False).head(8).reset_index()
    top_skus = [{"material": str(r["material"]), "desc": str(r["material_desc"]), "group": str(r["material_group"]), "value": float(r["monthly_purchase_value"])} for _, r in sku.iterrows()]
    groups = [{"name": str(k), "value": float(v)} for k, v in tg.head(8).items()]
    # category × month matrix (top 8 groups) — powers the heatmap + stacked stream
    mrows = []
    for g in list(tg.head(8).index):
        sub = mp[mp["material_group"] == g].groupby("month", observed=True)["monthly_purchase_value"].sum()
        vals = [float(sub.get(m, 0)) for m in order]
        mrows.append({"name": str(g), "values": vals, "total": float(sum(vals))})
    matrix = {"labels": [m[:3] for m in order], "months": order, "rows": mrows, "col_totals": total_series}
    # Category × month MARGIN matrix (MRP-proxy) aligned to the SAME rows/months, so the
    # heatmap can toggle Spend ↔ Margin %. GRN's own taxonomy differs from material_group,
    # so map via material → material_group. Margin % = (MRP − net price) on matched rows.
    margin_matrix = None
    try:
        g = da.filter_plant(da.load("fact_grn"), _plant(Plant)).copy()
        mat2grp = dict(zip(mp["material"].astype(str), mp["material_group"].astype(str)))
        g2 = g[(g["unit_mrp"] > 0) & (g["net_price"] > 0) & (g["gr_qty"] > 0)].copy()
        if not g2.empty:
            g2["grp"] = g2["material"].astype(str).map(mat2grp)
            g2["mrp_val"] = g2["gr_qty"] * g2["unit_mrp"]
            g2["cost_val"] = g2["gr_qty"] * g2["net_price"]
            piv_mrp = g2.groupby(["grp", "month"], observed=True)["mrp_val"].sum()
            piv_cost = g2.groupby(["grp", "month"], observed=True)["cost_val"].sum()
            mgn_rows = []
            for grp in list(tg.head(8).index):
                gk = str(grp); vals = []
                tmv = tcv = 0.0
                for m in order:
                    mv = float(piv_mrp.get((gk, m), 0.0)); cv = float(piv_cost.get((gk, m), 0.0))
                    tmv += mv; tcv += cv
                    vals.append(round((mv - cv) / mv * 100, 1) if mv > 0 else None)
                mgn_rows.append({"name": gk, "values": vals,
                                 "total": (round((tmv - tcv) / tmv * 100, 1) if tmv > 0 else None)})
            margin_matrix = {"labels": [m[:3] for m in order], "months": order, "rows": mgn_rows}
    except Exception:
        margin_matrix = None
    return {"totals": {"total": total, "avg": total / max(len(order), 1),
                       "top_group": str(top_groups[0]) if top_groups else "-", "skus": int(mp["material"].nunique())},
            "timeline": {"labels": [m[:3] for m in order], "total": total_series, "series": series},
            "matrix": matrix, "margin_matrix": margin_matrix, "top_skus": top_skus, "groups": groups}


# ---------------- MONTHLY PURCHASE VALUE (KPI_6) ----------------
@router.get("/kpi/monthly-purchase-value")
def monthly_purchase_value(Plant: str = Query(None), Material: str = Query(None), MaterialGroup: str = Query(None)):
    df = _filt(da.load("kpi_monthly_purchase_value"), Plant, Material, MaterialGroup)
    return _monthly(df, "monthly_purchase_value", "Monthly Purchase Value")


@router.get("/kpi/monthly-purchase-value-table")
def monthly_purchase_value_table(request: Request, Plant: str = Query(None)):
    return da.paginate("kpi_monthly_purchase_value", _plant(Plant), dict(request.query_params),
        col_map={"year": "year", "period": "month", "materialId": "material", "materialName": "material_desc",
                 "materialGroup": "material_group", "monthlyPurchaseValue": "monthly_purchase_value"},
        columns=["year", "month", "material", "material_desc", "material_group", "monthly_purchase_value"],
        rename={"year": "year", "month": "period", "material": "materialId", "material_desc": "materialName",
                "material_group": "materialGroup", "monthly_purchase_value": "monthlyPurchaseValue"})


# ---------------- VENDOR VOLUME vs MARGIN (KPI_7) — from GRN ----------------
def _vendor_margin(plant):
    grn = da.filter_plant(da.load("fact_grn"), _plant(plant)).copy()
    g = grn.groupby(["vendor_name", "year", "month"], as_index=False, observed=True).agg(
        grn_volume=("gr_qty", "sum"), unit_cost=("net_price", "mean"), unit_mrp=("unit_mrp", "mean"))
    g["margin"] = np.where(g["unit_mrp"] > 0, (g["unit_mrp"] - g["unit_cost"]) / g["unit_mrp"] * 100, 0)
    return g


@router.get("/kpi/vendor-volume-vs-margin")
def vendor_margin(Plant: str = Query(None)):
    g = _vendor_margin(Plant)
    g = g.sort_values("grn_volume", ascending=False).head(200)
    return [{"Year": int(r["year"]) if pd.notna(r["year"]) else None, "Month": r["month"], "Vendor Name": r["vendor_name"],
             "GRN Volume": float(r["grn_volume"]), "Unit Purchase Price": round(float(r["unit_cost"]), 2),
             "Unit MRP Price": round(float(r["unit_mrp"]), 2), "Margin (%)": round(float(r["margin"]), 1),
             "Margin": round(float(r["margin"]), 1)} for _, r in g.iterrows()]


@router.get("/kpi/vendor-volume-vs-margin-table")
def vendor_margin_table(request: Request, Plant: str = Query(None)):
    g = _vendor_margin(Plant)
    return _paginate_df(g, request, ["year", "month", "vendor_name", "grn_volume", "unit_cost", "unit_mrp", "margin"],
        {"year": "year", "month": "period", "vendor_name": "vendorName", "grn_volume": "grnVolume",
         "unit_cost": "averageUnitCost", "unit_mrp": "sellingPrice", "margin": "margin"})


# ---------------- UNITS CONSUMED (KPI_8) ----------------
@router.get("/kpi/unit-sold-per-sku")
def units_sold(Plant: str = Query(None), Material: str = Query(None), MaterialGroup: str = Query(None)):
    df = _filt(da.load("kpi_units_consumed"), Plant, Material, MaterialGroup)
    return _monthly(df, "total_units", "Total Units Sold")


@router.get("/kpi/unit-sold-per-sku-table")
def units_sold_table(request: Request, Plant: str = Query(None)):
    return da.paginate("kpi_units_consumed", _plant(Plant), dict(request.query_params),
        col_map={"year": "year", "period": "month", "materialId": "material", "materialName": "material_desc",
                 "materialGroup": "material_group", "totalUnitsSold": "total_units"},
        columns=["year", "month", "material", "material_desc", "material_group", "total_units"],
        rename={"year": "year", "month": "period", "material": "materialId", "material_desc": "materialName",
                "material_group": "materialGroup", "total_units": "totalUnitsSold"})


# ---------------- REVENUE DISTRIBUTION (KPI_9) — MRP proxy ----------------
@router.get("/kpi/revenue-distribution")
def revenue_distribution(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_units_consumed"), _plant(Plant))
    g = df.groupby(["year", "month"], as_index=False, observed=True)["consumption_cost"].sum()
    g["_mk"] = g["month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    g = g.sort_values(["year", "_mk"])
    return [{"Year": int(r["year"]), "Month": r["month"], "OP Sales": 0.0, "IP Issue": 0.0,
             "Internal Consumption": float(r["consumption_cost"])} for _, r in g.iterrows()]


@router.get("/kpi/revenue-distribution-table")
def revenue_distribution_table(request: Request, Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_units_consumed"), _plant(Plant)).copy()
    g = df.groupby(["year", "month"], as_index=False, observed=True)["consumption_cost"].sum()
    g["op"] = 0.0; g["ip"] = 0.0
    return _paginate_df(g, request, ["year", "month", "op", "ip", "consumption_cost"],
        {"year": "year", "month": "period", "op": "opSales", "ip": "ipIssue", "consumption_cost": "internalConsumption"})


# ---------------- REVENUE PER STORAGE LOCATION — proxy ----------------
@router.get("/kpi/revenue-per-storage-location")
def revenue_per_location(Plant: str = Query(None)):
    df = da.filter_plant(da.load("kpi_units_consumed"), _plant(Plant))
    g = df.groupby(["year", "month"], as_index=False, observed=True)["consumption_cost"].sum()
    g["_mk"] = g["month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    g = g.sort_values(["year", "_mk"])
    return [{"Year": int(r["year"]), "Month": r["month"], "Amount in LC": float(r["consumption_cost"])} for _, r in g.iterrows()]


# ================= CONSUMPTION & REVENUE insights (C8/C9) =================
def _msort(df):
    df = df.copy(); df["_k"] = df["month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    return df.sort_values(["year", "_k"])


def _chrono_months(df):
    g = _msort(df.groupby(["year", "month"], observed=True).size().reset_index())
    return list(dict.fromkeys(g["month"].astype(str)))


@router.get("/portfolio/consumption/overview")
def consumption_overview(Plant: str = Query(None)):
    return _consumption_overview_cached(_plant(Plant))


@_cache
def _consumption_overview_cached(pl):
    uc = da.filter_plant(da.load("kpi_units_consumed"), pl).copy()
    dp = da.filter_plant(da.load("kpi_consumption_by_department"), pl).copy()
    cost = float(uc["consumption_cost"].sum()); units = float(uc["total_units"].sum())
    m = _msort(uc.groupby(["year", "month"], observed=True).agg(cost=("consumption_cost", "sum"), units=("total_units", "sum")).reset_index())
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "cost": float(r["cost"]), "units": float(r["units"])} for _, r in m.iterrows()]
    last_mom = ((timeline[-1]["cost"] - timeline[-2]["cost"]) / timeline[-2]["cost"] * 100) if len(timeline) >= 2 and timeline[-2]["cost"] else 0.0
    uc["cat"] = uc["material_group"].fillna("Uncategorized").astype(str)
    cg = uc.groupby("cat", observed=True).agg(value=("consumption_cost", "sum"), units=("total_units", "sum")).reset_index().sort_values("value", ascending=False)
    categories = [{"name": str(r["cat"]), "value": float(r["value"]), "units": float(r["units"]),
                   "uncat": str(r["cat"]).strip().lower() in ("uncategorized", "", "nan", "none")} for _, r in cg.head(10).iterrows()]
    dg = dp.groupby("cost_ctr", observed=True).agg(value=("consumption_cost", "sum"), qty=("consumption_qty", "sum")).reset_index().sort_values("value", ascending=False)
    dtot = float(dg["value"].sum())
    departments = [{"code": str(r["cost_ctr"]), "value": float(r["value"]), "qty": float(r["qty"]), "share": float(r["value"] / dtot * 100) if dtot else 0.0} for _, r in dg.head(8).iterrows()]
    dept_top5 = float(dg.head(5)["value"].sum() / dtot * 100) if dtot else 0.0
    sg = uc.groupby(["material", "material_desc"], observed=True).agg(units=("total_units", "sum"), cost=("consumption_cost", "sum")).reset_index().sort_values("cost", ascending=False)
    skus = [{"material": str(r["material"]), "desc": str(r["material_desc"]), "units": float(r["units"]), "cost": float(r["cost"])} for _, r in sg.head(8).iterrows()]
    cards = {"unit-sold-per-sku": {"value": units, "kind": "num", "sub": "units consumed"},
             "consumption-by-department": {"value": cost, "kind": "inr", "sub": "6-mo cost"}}
    return {"totals": {"cost": cost, "units": units, "materials": int(uc["material"].nunique()),
                       "departments": int(dp["cost_ctr"].nunique()), "plants": int(uc["plant"].nunique()),
                       "avg_month": cost / max(len(timeline), 1), "last_mom": last_mom, "dept_top5": dept_top5},
            "timeline": timeline, "categories": categories, "departments": departments, "skus": skus, "cards": cards}


@router.get("/kpi/unit-sold-per-sku/insights")
def units_consumed_insights(Plant: str = Query(None)):
    uc = da.filter_plant(da.load("kpi_units_consumed"), _plant(Plant)).copy()
    units = float(uc["total_units"].sum()); cost = float(uc["consumption_cost"].sum())
    m = _msort(uc.groupby(["year", "month"], observed=True).agg(units=("total_units", "sum"), cost=("consumption_cost", "sum")).reset_index())
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "units": float(r["units"]), "cost": float(r["cost"])} for _, r in m.iterrows()]
    uc["cat"] = uc["material_group"].fillna("Uncategorized").astype(str)
    cg = uc.groupby("cat", observed=True).agg(units=("total_units", "sum"), cost=("consumption_cost", "sum")).reset_index().sort_values("units", ascending=False)
    ctot = float(cg["units"].sum())
    categories = [{"name": str(r["cat"]), "units": float(r["units"]), "cost": float(r["cost"]), "share": float(r["units"] / ctot * 100) if ctot else 0.0,
                   "uncat": str(r["cat"]).strip().lower() in ("uncategorized", "", "nan", "none")} for _, r in cg.head(9).iterrows()]
    sg = uc.groupby(["material", "material_desc"], observed=True).agg(units=("total_units", "sum"), cost=("consumption_cost", "sum")).reset_index()
    sg["cpu"] = np.where(sg["units"] > 0, sg["cost"] / sg["units"], 0.0)
    mk = lambda df: [{"material": str(r["material"]), "desc": str(r["material_desc"]), "units": float(r["units"]), "cost": float(r["cost"]), "cpu": float(r["cpu"])} for _, r in df.iterrows()]
    top_units = mk(sg.sort_values("units", ascending=False).head(10))
    scatter = mk(sg[sg["units"] >= 1].sort_values("cost", ascending=False).head(45))
    # category × month matrix (top 8 groups by units) — powers the heatmap
    order = _chrono_months(uc)
    top_cats = list(cg.head(8)["cat"])
    mrows = []
    for c in top_cats:
        g = uc[uc["cat"] == c].groupby("month", observed=True).agg(u=("total_units", "sum"), co=("consumption_cost", "sum"))
        vals = [float(g["u"].get(m, 0)) for m in order]; costs = [float(g["co"].get(m, 0)) for m in order]
        mrows.append({"name": str(c), "values": vals, "cost": costs, "total": float(sum(vals)), "cost_total": float(sum(costs)),
                      "uncat": str(c).strip().lower() in ("uncategorized", "", "nan", "none")})
    matrix = {"labels": [m[:3] for m in order], "months": order, "rows": mrows}
    return {"totals": {"units": units, "cost": cost, "materials": int(uc["material"].nunique()),
                       "cpu": (cost / units) if units else 0.0, "top_material": top_units[0]["desc"] if top_units else "-"},
            "timeline": timeline, "categories": categories, "skus": top_units, "scatter": scatter, "matrix": matrix}


@router.get("/kpi/consumption-by-department/insights")
def dept_insights(Plant: str = Query(None)):
    dp = da.filter_plant(da.load("kpi_consumption_by_department"), _plant(Plant)).copy()
    cost = float(dp["consumption_cost"].sum()); qty = float(dp["consumption_qty"].sum())
    dg = dp.groupby("cost_ctr", observed=True).agg(value=("consumption_cost", "sum"), qty=("consumption_qty", "sum")).reset_index().sort_values("value", ascending=False)
    tot = float(dg["value"].sum()); n = int(len(dg))
    dg["share"] = np.where(tot > 0, dg["value"] / tot * 100, 0.0); dg["cum"] = dg["share"].cumsum()
    n80 = int((dg["cum"] <= 80).sum()) + 1
    departments = [{"code": str(r["cost_ctr"]), "value": float(r["value"]), "qty": float(r["qty"]), "share": float(r["share"]), "cum": float(r["cum"])} for _, r in dg.head(24).iterrows()]
    m = _msort(dp.groupby(["year", "month"], observed=True).agg(cost=("consumption_cost", "sum"), qty=("consumption_qty", "sum")).reset_index())
    timeline = [{"label": str(r["month"])[:3], "month": str(r["month"]), "cost": float(r["cost"]), "qty": float(r["qty"])} for _, r in m.iterrows()]
    # top-6 department × month matrix — powers the stacked stream
    order = _chrono_months(dp)
    srows = []
    for code in list(dg.head(6)["cost_ctr"]):
        sub = dp[dp["cost_ctr"] == code].groupby("month", observed=True)["consumption_cost"].sum()
        srows.append({"name": str(code), "values": [float(sub.get(mm, 0)) for mm in order]})
    matrix = {"labels": [mm[:3] for mm in order], "months": order, "rows": srows}
    return {"totals": {"cost": cost, "qty": qty, "departments": n, "top_dept": str(dg["cost_ctr"].iloc[0]) if n else "-",
                       "top1": float(dg["share"].iloc[0]) if n else 0.0, "top5": float(dg.head(5)["share"].sum()),
                       "top10": float(dg.head(10)["share"].sum()), "n80": n80, "hhi": float((dg["share"] ** 2).sum())},
            "departments": departments, "timeline": timeline, "matrix": matrix}


# ================= FORECASTING portfolio overview (D1–D8) =================
@router.get("/portfolio/forecasting/overview")
def forecasting_overview(Plant: str = Query(None)):
    return _forecasting_overview_cached(_plant(Plant))


@_cache
def _forecasting_overview_cached(pl):
    fs = da.filter_plant(da.load("forecast_sales"), pl).copy()
    rp = da.filter_plant(da.load("stock_replenishment_and_aging_risk"), pl).copy()
    rd = da.filter_plant(da.load("kpi_stock_radar"), pl).copy()
    ar = da.filter_plant(da.load("kpi_aging_risk_forecast"), pl).copy()
    acc = da.load("forecast_accuracy")
    accd = {str(r["metric"]): float(r["value"]) for _, r in acc.iterrows()}

    fs["pd"] = pd.to_datetime(fs["posting_date"], errors="coerce")
    g = fs.groupby("pd").agg(actual=("sales_quantity", "sum"), fc=("sales_quantity_forecast", "sum"),
                             lo=("lower_bound_sales_quantity_forecast", "sum"), hi=("upper_bound_sales_quantity_forecast", "sum"),
                             cf=("cashflow_forecast", "sum"), cflo=("lower_bound_cashflow_forecast", "sum"),
                             cfhi=("upper_bound_cashflow_forecast", "sum")).reset_index().sort_values("pd")
    timeline = []
    for _, r in g.iterrows():
        isf = float(r["fc"]) > 0
        timeline.append({"label": r["pd"].strftime("%b"), "month": r["pd"].strftime("%b %Y"),
                         "actual": float(r["actual"]) if float(r["actual"]) > 0 else None,
                         "forecast": float(r["fc"]) if isf else None,
                         "lower": float(r["lo"]) if isf else None, "upper": float(r["hi"]) if isf else None,
                         "is_forecast": bool(isf)})
    fcr = [t for t in timeline if t["is_forecast"]]
    cashflow = [{"label": r["pd"].strftime("%b"), "forecast": float(r["cf"]), "lower": float(r["cflo"]), "upper": float(r["cfhi"])}
                for _, r in g.iterrows() if float(r["cf"]) > 0]

    need = rp[rp["replenishment_quantity"] > 0].copy()
    need["val"] = need["replenishment_quantity"] * need["unit_cost"]
    replen_skus = int(len(need)); replen_qty = float(need["replenishment_quantity"].sum()); replen_value = float(need["val"].sum())
    top = need.sort_values("val", ascending=False).head(8)
    top_reorder = [{"material": str(r["material_id"]), "desc": str(r.get("material_desc", "")), "qty": float(r["replenishment_quantity"]),
                    "value": float(r["val"]), "cover": float(r.get("demand_monthly", 0))} for _, r in top.iterrows()]

    radar = [{"status": str(k), "count": int(v)} for k, v in rd["radar_status"].value_counts().items()] if "radar_status" in rd else []
    aging = [{"status": str(k), "count": int(v)} for k, v in ar["aging_risk_forecast"].value_counts().items()] if "aging_risk_forecast" in ar else []

    cards = {
        "expected-demand": {"value": fcr[0]["forecast"] if fcr else 0.0, "kind": "num", "sub": "next-month units"},
        "cash-flow-forecast": {"value": cashflow[0]["forecast"] if cashflow else 0.0, "kind": "inr", "sub": "next-month spend"},
        "stock-replenishment": {"value": float(replen_skus), "kind": "num", "sub": "SKUs to reorder"},
        "fulfillment-rate": {"value": accd.get("Aggregate Forecast Accuracy %", 0.0), "kind": "pct", "sub": "forecast accuracy"},
        "stock-radar": {"value": float(sum(x["count"] for x in radar if x["status"] == "Stock-Out Risk")), "kind": "num", "sub": "stock-out risk SKUs"},
        "aging-risk-forecast": {"value": float(sum(x["count"] for x in aging if x["status"] == "Rising")), "kind": "num", "sub": "rising-risk SKUs"},
    }
    return {"totals": {"next_demand": fcr[0]["forecast"] if fcr else 0.0, "next_lower": fcr[0]["lower"] if fcr else 0.0, "next_upper": fcr[0]["upper"] if fcr else 0.0,
                       "accuracy": accd.get("Aggregate Forecast Accuracy %", 0.0), "weighted_acc": accd.get("Weighted Forecast Accuracy %", 0.0),
                       "mape": accd.get("Weighted MAPE %", 0.0), "series": int(accd.get("Series count", 0)),
                       "replen_skus": replen_skus, "replen_qty": replen_qty, "replen_value": replen_value,
                       "cashflow_next": cashflow[0]["forecast"] if cashflow else 0.0, "horizon": len(fcr)},
            "timeline": timeline, "cashflow": cashflow, "radar": radar, "aging": aging, "top_reorder": top_reorder, "cards": cards}


def _clean_group(g) -> str:
    g = str(g).strip()
    if not g or g.lower() in ("nan", "none"):
        return "Uncategorised"
    g = g.split("-", 1)[-1] if "-" in g else g
    return g.strip().title() or "Uncategorised"


@router.get("/forecast/demand-insights")
def demand_insights(Plant: str = Query(None)):
    """Rich demand-forecast insights for the bespoke Expected-Usage page:
    aggregate cone timeline, headline totals, top items by forecast demand and a
    per-category demand breakdown."""
    pl = _plant(Plant)
    fs = da.filter_plant(da.load("forecast_sales"), pl).copy()
    accd = {str(r["metric"]): float(r["value"]) for _, r in da.load("forecast_accuracy").iterrows()}
    dm = da.load("dim_material")[["material", "material_desc", "material_group"]].copy()
    dm["material"] = dm["material"].astype(str)
    desc_of = dict(zip(dm["material"], dm["material_desc"].astype(str)))
    grp_of = dict(zip(dm["material"], dm["material_group"].astype(str)))

    fs["material_id"] = fs["material_id"].astype(str)
    fs["pd"] = pd.to_datetime(fs["posting_date"], errors="coerce")

    g = (fs.groupby("pd").agg(actual=("sales_quantity", "sum"), fc=("sales_quantity_forecast", "sum"),
                              lo=("lower_bound_sales_quantity_forecast", "sum"),
                              hi=("upper_bound_sales_quantity_forecast", "sum")).reset_index().sort_values("pd"))
    timeline = []
    for _, r in g.iterrows():
        isf = float(r["fc"]) > 0
        timeline.append({"label": r["pd"].strftime("%b"), "month": r["pd"].strftime("%b %Y"),
                         "actual": float(r["actual"]) if float(r["actual"]) > 0 else None,
                         "forecast": float(r["fc"]) if isf else None,
                         "lower": float(r["lo"]) if isf else None, "upper": float(r["hi"]) if isf else None,
                         "is_forecast": bool(isf)})
    fcr = [t for t in timeline if t["is_forecast"]]

    ff = fs[fs["sales_quantity_forecast"].notna() & (fs["sales_quantity_forecast"] > 0)].copy()
    last_actual = (fs[fs["sales_quantity"].notna()].sort_values("pd")
                     .groupby("material_id")["sales_quantity"].last().to_dict())
    itm = ff.groupby("material_id").agg(fc=("sales_quantity_forecast", "sum"),
                                        lo=("lower_bound_sales_quantity_forecast", "sum"),
                                        hi=("upper_bound_sales_quantity_forecast", "sum"),
                                        next_mo=("sales_quantity_forecast", "first")).reset_index()
    top = itm.nlargest(12, "fc")
    top_items = [{"material": r["material_id"], "desc": desc_of.get(r["material_id"], r["material_id"]),
                  "group": _clean_group(grp_of.get(r["material_id"], "")), "forecast": float(r["fc"]),
                  "next_mo": float(r["next_mo"]), "lower": float(r["lo"]), "upper": float(r["hi"]),
                  "last_actual": float(last_actual.get(r["material_id"], 0.0))} for _, r in top.iterrows()]

    ff["grp"] = ff["material_id"].map(grp_of).fillna("")
    cat = ff[ff["grp"].astype(str).str.len() > 0].groupby("grp")["sales_quantity_forecast"].sum().reset_index()
    cat = cat.sort_values("sales_quantity_forecast", ascending=False).head(10)
    by_category = [{"group": _clean_group(r["grp"]), "forecast": float(r["sales_quantity_forecast"])} for _, r in cat.iterrows()]

    return {"timeline": timeline,
            "totals": {"next_demand": fcr[0]["forecast"] if fcr else 0.0,
                       "next_lower": fcr[0]["lower"] if fcr else 0.0, "next_upper": fcr[0]["upper"] if fcr else 0.0,
                       "total_horizon": float(ff["sales_quantity_forecast"].sum()),
                       "accuracy": accd.get("Aggregate Forecast Accuracy %", 0.0),
                       "weighted_acc": accd.get("Weighted Forecast Accuracy %", 0.0),
                       "materials": int(ff["material_id"].nunique()), "horizon": len(fcr)},
            "top_items": top_items, "by_category": by_category}


@router.get("/forecast/cashflow-insights")
def cashflow_insights(Plant: str = Query(None)):
    """Rich procurement-budget (cash-flow) insights for the bespoke Cash-Flow page:
    monthly spend timeline (actual consumption cost -> forecast budget), headline
    rupee totals, biggest spend items and a per-category budget breakdown. Values
    are the forecast consumption cost (amount_lc) = the cash needed to restock."""
    pl = _plant(Plant)
    fs = da.filter_plant(da.load("forecast_sales"), pl).copy()
    accd = {str(r["metric"]): float(r["value"]) for _, r in da.load("forecast_accuracy").iterrows()}
    dm = da.load("dim_material")[["material", "material_desc", "material_group"]].copy()
    dm["material"] = dm["material"].astype(str)
    desc_of = dict(zip(dm["material"], dm["material_desc"].astype(str)))
    grp_of = dict(zip(dm["material"], dm["material_group"].astype(str)))

    fs["material_id"] = fs["material_id"].astype(str)
    fs["pd"] = pd.to_datetime(fs["posting_date"], errors="coerce")

    g = (fs.groupby("pd").agg(actual=("sales_value", "sum"), fc=("cashflow_forecast", "sum"),
                              lo=("lower_bound_cashflow_forecast", "sum"),
                              hi=("upper_bound_cashflow_forecast", "sum")).reset_index().sort_values("pd"))
    timeline, cum = [], 0.0
    for _, r in g.iterrows():
        isf = float(r["fc"]) > 0
        if isf:
            cum += float(r["fc"])
        timeline.append({"label": r["pd"].strftime("%b"), "month": r["pd"].strftime("%b %Y"),
                         "actual": float(r["actual"]) if float(r["actual"]) > 0 else None,
                         "forecast": float(r["fc"]) if isf else None,
                         "lower": float(r["lo"]) if isf else None, "upper": float(r["hi"]) if isf else None,
                         "cumulative": cum if isf else None, "is_forecast": bool(isf)})
    fcr = [t for t in timeline if t["is_forecast"]]

    ff = fs[fs["cashflow_forecast"].notna() & (fs["cashflow_forecast"] > 0)].copy()
    last_actual = (fs[fs["sales_value"].notna()].sort_values("pd")
                     .groupby("material_id")["sales_value"].last().to_dict())
    itm = ff.groupby("material_id").agg(fc=("cashflow_forecast", "sum"),
                                        lo=("lower_bound_cashflow_forecast", "sum"),
                                        hi=("upper_bound_cashflow_forecast", "sum"),
                                        next_mo=("cashflow_forecast", "first")).reset_index()
    top = itm.nlargest(12, "fc")
    top_items = [{"material": r["material_id"], "desc": desc_of.get(r["material_id"], r["material_id"]),
                  "group": _clean_group(grp_of.get(r["material_id"], "")), "forecast": float(r["fc"]),
                  "next_mo": float(r["next_mo"]), "lower": float(r["lo"]), "upper": float(r["hi"]),
                  "last_actual": float(last_actual.get(r["material_id"], 0.0))} for _, r in top.iterrows()]

    ff["grp"] = ff["material_id"].map(grp_of).fillna("")
    cat = ff[ff["grp"].astype(str).str.len() > 0].groupby("grp")["cashflow_forecast"].sum().reset_index()
    cat = cat.sort_values("cashflow_forecast", ascending=False).head(10)
    by_category = [{"group": _clean_group(r["grp"]), "forecast": float(r["cashflow_forecast"])} for _, r in cat.iterrows()]

    total_h = float(ff["cashflow_forecast"].sum())
    return {"timeline": timeline,
            "totals": {"next_budget": fcr[0]["forecast"] if fcr else 0.0,
                       "next_lower": fcr[0]["lower"] if fcr else 0.0, "next_upper": fcr[0]["upper"] if fcr else 0.0,
                       "total_horizon": total_h, "avg_month": (total_h / len(fcr)) if fcr else 0.0,
                       "accuracy": accd.get("Aggregate Forecast Accuracy %", 0.0),
                       "weighted_acc": accd.get("Weighted Forecast Accuracy %", 0.0),
                       "materials": int(ff["material_id"].nunique()), "horizon": len(fcr)},
            "top_items": top_items, "by_category": by_category}


def _replen_frame(pl):
    rp = da.filter_plant(da.load("stock_replenishment_and_aging_risk"), pl).copy()
    for c in ["closing_stock", "closing_stock_value", "demand_forecast", "demand_monthly",
              "unit_cost", "replenishment_quantity", "aging_days", "safe_stock"]:
        if c in rp:
            rp[c] = pd.to_numeric(rp[c], errors="coerce").fillna(0.0)
    rp["material_id"] = rp["material_id"].astype(str)
    rp["reorder_value"] = rp["replenishment_quantity"] * rp["unit_cost"]
    rp["cover"] = np.where(rp["demand_monthly"] > 0, rp["closing_stock"] / rp["demand_monthly"],
                           np.where(rp["closing_stock"] > 0, 999.0, 0.0))

    def _status(r):
        if r["closing_stock"] <= 0 and r["demand_forecast"] > 0:
            return "Stock-out"
        if r["aging_days"] > 365 and r["closing_stock"] > 0:
            return "Dead stock"
        if r["replenishment_quantity"] > 0 and r["cover"] < 1:
            return "Reorder now"
        if r["cover"] > 9 or r["aging_days"] > 180:
            return "Overstocked"
        return "Healthy"
    rp["status"] = rp.apply(_status, axis=1)
    return rp


@router.get("/forecast/replenishment-insights")
def replenishment_insights(Plant: str = Query(None)):
    """Reorder & aging-risk action board: a stock-health spectrum (understock ->
    overstock), the biggest items to reorder now, the items sitting too long, and
    a ladder of the cash locked in aging stock."""
    pl = _plant(Plant)
    rp = _replen_frame(pl)
    accd = {str(r["metric"]): float(r["value"]) for _, r in da.load("forecast_accuracy").iterrows()}

    ORDER = ["Stock-out", "Reorder now", "Healthy", "Overstocked", "Dead stock"]
    spectrum = [{"status": s, "count": int((rp["status"] == s).sum()),
                 "value": float(rp[rp["status"] == s]["closing_stock_value"].sum())} for s in ORDER]

    need = rp[rp["replenishment_quantity"] > 0].sort_values("reorder_value", ascending=False).head(8)
    order_now = [{"material": r["material_id"], "desc": str(r.get("material_desc", "")),
                  "group": _clean_group(r.get("material_group", "")), "qty": float(r["replenishment_quantity"]),
                  "value": float(r["reorder_value"]), "cover": float(r["cover"]), "stock": float(r["closing_stock"])}
                 for _, r in need.iterrows()]

    aged = rp[(rp["aging_days"] > 180) & (rp["closing_stock_value"] > 0)].sort_values("closing_stock_value", ascending=False).head(8)
    aging = [{"material": r["material_id"], "desc": str(r.get("material_desc", "")),
              "group": _clean_group(r.get("material_group", "")), "value": float(r["closing_stock_value"]),
              "aging_days": int(r["aging_days"]), "bucket": str(r["aging_risk"])} for _, r in aged.iterrows()]

    LAD = ["<3 Months", "3-6 Months", "6-12 Months", "1+ Year"]
    ladder = [{"bucket": b, "value": float(rp[rp["aging_risk"] == b]["closing_stock_value"].sum()),
               "count": int((rp["aging_risk"] == b).sum())} for b in LAD]

    # reorder pressure by category — the hero chart (clean, actionable columns):
    # how much reorder cash each department needs vs cash tied up in its aging stock.
    cat = rp.copy()
    cat["g"] = cat["material_group"].apply(_clean_group)
    cat["aged_val"] = np.where(cat["aging_days"] > 180, cat["closing_stock_value"], 0.0)
    by = (cat.groupby("g").agg(reorder_value=("reorder_value", "sum"),
                               reorder_count=("replenishment_quantity", lambda s: int((s > 0).sum())),
                               aging_value=("aged_val", "sum")).reset_index()
             .sort_values("reorder_value", ascending=False).head(7))
    by_category = [{"group": r["g"], "reorder_value": float(r["reorder_value"]),
                    "reorder_count": int(r["reorder_count"]), "aging_value": float(r["aging_value"])}
                   for _, r in by.iterrows()]

    totals = {"reorder_skus": int((rp["replenishment_quantity"] > 0).sum()),
              "reorder_value": float(rp["reorder_value"].sum()),
              "reorder_qty": float(rp["replenishment_quantity"].sum()),
              "stockout_skus": int((rp["status"] == "Stock-out").sum()),
              "aging_skus": int((rp["aging_days"] > 180).sum()),
              "aging_value": float(rp[rp["aging_days"] > 180]["closing_stock_value"].sum()),
              "healthy_skus": int((rp["status"] == "Healthy").sum()),
              "total_skus": int(len(rp)), "stock_value": float(rp["closing_stock_value"].sum()),
              "accuracy": accd.get("Aggregate Forecast Accuracy %", 0.0)}
    return {"totals": totals, "spectrum": spectrum, "by_category": by_category,
            "order_now": order_now, "aging": aging, "ladder": ladder}


@router.get("/revenue/items")
def revenue_items(group: str = Query(None), manufacturer: str = Query(None), hospital: str = Query(None),
                  sort: str = Query("revenue"), limit: int = Query(400)):
    """Full billed-item list (with true margin) for the Revenue & Margin drill-down —
    filterable by category, or by manufacturer / hospital (material×dimension cross-tabs)."""
    if manufacturer:
        fp = os.path.join(_KPI_DIR, "sales_by_material_mfr.parquet")
        m = pd.read_parquet(fp).copy() if os.path.exists(fp) else pd.DataFrame()
        if len(m):
            m = m[m["manufacturer"].astype(str) == str(manufacturer)].copy()
    elif hospital:
        fp = os.path.join(_KPI_DIR, "sales_by_material_hospital.parquet")
        m = pd.read_parquet(fp).copy() if os.path.exists(fp) else pd.DataFrame()
        if len(m):
            m = m[m["hospital"].astype(str) == str(hospital)].copy()
    else:
        fp = os.path.join(_KPI_DIR, "sales_by_material.parquet")
        m = pd.read_parquet(fp).copy() if os.path.exists(fp) else pd.DataFrame()
    if not len(m):
        return {"count": 0, "returned": 0, "items": []}
    m["margin"] = m["revenue"] - m["cost"]
    m["g"] = m["group"].apply(_clean_group)
    if group and not manufacturer and not hospital:
        m = m[m["g"] == group]
    m["margin_pct"] = np.where(m["revenue"] > 0, m["margin"] / m["revenue"] * 100, 0.0)
    sortcol = sort if sort in ("revenue", "margin", "margin_pct", "qty") else "revenue"
    total = int(len(m))
    m = m.sort_values(sortcol, ascending=False).head(int(limit))
    items = [{"material": str(r["material"]), "desc": str(r["desc"]), "group": r["g"],
              "revenue": float(r["revenue"]), "margin": float(r["margin"]),
              "margin_pct": float(r["margin_pct"]), "qty": float(r["qty"])} for _, r in m.iterrows()]
    return {"count": total, "returned": len(items), "items": items}


@router.get("/forecast/risk-items")
def risk_items(Plant: str = Query(None), status: str = Query(None), aging: str = Query(None),
               kind: str = Query(None), limit: int = Query(200)):
    """Full item list behind any reorder/aging cut — powers the drill-downs the
    client asked for (click a status / bucket / 'see all' → the actual items)."""
    rp = _replen_frame(_plant(Plant))
    if status:
        sub = rp[rp["status"] == status]
    elif aging:
        sub = rp[rp["aging_risk"] == aging]
    elif kind == "order_now":
        sub = rp[rp["replenishment_quantity"] > 0]
    elif kind == "aging":
        sub = rp[(rp["aging_days"] > 180) & (rp["closing_stock_value"] > 0)]
    else:
        sub = rp
    reorder_sorted = bool(kind == "order_now" or status in ("Stock-out", "Reorder now"))
    sortcol = "reorder_value" if reorder_sorted else "closing_stock_value"
    total = int(len(sub))
    sub = sub.sort_values(sortcol, ascending=False).head(int(limit))
    items = [{"material": str(r["material_id"]), "desc": str(r.get("material_desc", "")),
              "group": _clean_group(r.get("material_group", "")), "status": str(r["status"]),
              "stock": float(r["closing_stock"]), "stock_value": float(r["closing_stock_value"]),
              "demand_monthly": float(r["demand_monthly"]), "cover": float(r["cover"]),
              "reorder_qty": float(r["replenishment_quantity"]), "reorder_value": float(r["reorder_value"]),
              "aging_days": int(r["aging_days"])} for _, r in sub.iterrows()]
    return {"count": total, "returned": len(items), "items": items}


@router.get("/forecast/item-risk")
def item_risk(Plant: str = Query(None), Material: str = Query(...)):
    """Single-SKU reorder & aging status for the 'check any item' panel."""
    rp = _replen_frame(_plant(Plant))
    sub = rp[rp["material_id"] == str(Material)]
    if not len(sub):
        return {"found": False}
    r = sub.iloc[0]
    return {"found": True, "material": str(r["material_id"]), "desc": str(r.get("material_desc", "")),
            "group": _clean_group(r.get("material_group", "")), "status": str(r["status"]),
            "stock": float(r["closing_stock"]), "stock_value": float(r["closing_stock_value"]),
            "demand_monthly": float(r["demand_monthly"]), "demand_forecast": float(r["demand_forecast"]),
            "cover": float(r["cover"]), "safe_stock": float(r["safe_stock"]),
            "reorder_qty": float(r["replenishment_quantity"]), "reorder_value": float(r["reorder_value"]),
            "aging_days": int(r["aging_days"]), "bucket": str(r["aging_risk"]), "unit_cost": float(r["unit_cost"])}


@router.get("/revenue/insights")
def revenue_insights():
    """Real revenue & margin from IP + OP billing (fact_sales aggregates).
    Revenue = billed MRP, Margin = MRP − cost (actual, not proxy). Splits by
    patient (IP/OP), hospital, manufacturer, category and top items."""
    def _p(name):
        fp = os.path.join(_KPI_DIR, name + ".parquet")
        return pd.read_parquet(fp) if os.path.exists(fp) else None
    tot = _p("sales_totals")
    if tot is None or not len(tot):
        return {"ready": False}
    mon = _p("sales_monthly"); hos = _p("sales_by_hospital")
    mfr = _p("sales_by_manufacturer"); matx = _p("sales_by_material")

    def rc(df):
        return (float(df["revenue"].sum()), float(df["cost"].sum()), float(df["qty"].sum()), int(df["lines"].sum()))
    ipr, ipc, ipq, ipl = rc(tot[tot.patient == "IP"]) if (tot.patient == "IP").any() else (0, 0, 0, 0)
    opr, opc, opq, opl = rc(tot[tot.patient == "OP"]) if (tot.patient == "OP").any() else (0, 0, 0, 0)
    rev, cost = ipr + opr, ipc + opc
    margin = rev - cost

    timeline = []
    if mon is not None and len(mon):
        mon["month"] = mon["month"].astype(str)
        for mm in sorted(mon["month"].unique()):
            sub = mon[mon.month == mm]
            ipx, opx = sub[sub.patient == "IP"], sub[sub.patient == "OP"]
            ir, ic = float(ipx.revenue.sum()), float(ipx.cost.sum())
            orr, oc = float(opx.revenue.sum()), float(opx.cost.sum())
            timeline.append({"month": mm, "label": _MN.get(mm[5:7], mm), "ip_revenue": ir, "op_revenue": orr,
                             "revenue": ir + orr, "ip_margin": ir - ic, "op_margin": orr - oc, "margin": (ir + orr) - (ic + oc)})

    def top(df, n, namecol):
        if df is None or not len(df):
            return []
        d = df.copy(); d["margin"] = d["revenue"] - d["cost"]
        d = d.sort_values("revenue", ascending=False).head(n)
        return [{namecol: str(r[namecol]), "revenue": float(r.revenue), "margin": float(r.margin),
                 "qty": float(r.qty), "margin_pct": (float(r.margin) / float(r.revenue) * 100 if r.revenue else 0.0)}
                for _, r in d.iterrows()]
    by_hospital = top(hos, 8, "hospital")
    by_manufacturer = top(mfr, 10, "manufacturer")

    top_items, by_category = [], []
    if matx is not None and len(matx):
        mm = matx.copy(); mm["margin"] = mm["revenue"] - mm["cost"]
        for _, r in mm.sort_values("revenue", ascending=False).head(12).iterrows():
            top_items.append({"material": str(r.material), "desc": str(r.desc), "group": _clean_group(r.group),
                              "revenue": float(r.revenue), "margin": float(r.margin), "qty": float(r.qty),
                              "margin_pct": (float(r.margin) / float(r.revenue) * 100 if r.revenue else 0.0)})
        mm["g"] = mm["group"].apply(_clean_group)
        cat = mm.groupby("g").agg(revenue=("revenue", "sum"), cost=("cost", "sum")).reset_index()
        cat["margin"] = cat.revenue - cat.cost
        for _, r in cat.sort_values("revenue", ascending=False).head(8).iterrows():
            by_category.append({"group": r.g, "revenue": float(r.revenue), "margin": float(r.margin),
                                "margin_pct": (float(r.margin) / float(r.revenue) * 100 if r.revenue else 0.0)})

    try:
        internal = float(da.load("fact_consumption")["amount_lc"].sum())
    except Exception:
        internal = 0.0

    return {"ready": True,
            "totals": {"revenue": rev, "cost": cost, "margin": margin, "margin_pct": (margin / rev * 100 if rev else 0.0),
                       "ip_revenue": ipr, "op_revenue": opr, "ip_margin": ipr - ipc, "op_margin": opr - opc,
                       "ip_share": (ipr / rev * 100 if rev else 0.0), "op_share": (opr / rev * 100 if rev else 0.0),
                       "qty": ipq + opq, "lines": ipl + opl,
                       "materials": int(matx.material.nunique()) if matx is not None else 0,
                       "manufacturers": int(len(mfr)) if mfr is not None else 0,
                       "hospitals": int(len(hos)) if hos is not None else 0,
                       "internal_cost": internal, "months": len(timeline)},
            "timeline": timeline, "by_hospital": by_hospital, "by_manufacturer": by_manufacturer,
            "top_items": top_items, "by_category": by_category}


# ---------------- helpers ----------------
def _num(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def _paginate_df(df: pd.DataFrame, request: Request, columns, rename):
    """Paginate an in-memory df (filter protocol + sort + page) -> {data,total}."""
    params = dict(request.query_params)
    df = da._apply_filter_protocol(df.copy(), params, {v: k for k, v in rename.items()})
    total = len(df)
    sf = params.get("sort_field"); so = (params.get("sort_order") or "asc").lower()
    src = {v: k for k, v in rename.items()}.get(sf, sf)
    if src and src in df.columns:
        df = df.sort_values(src, ascending=(so != "desc"), kind="mergesort", na_position="last")
    page = int(params.get("page", 0) or 0); size = int(params.get("page_size", 25) or 25)
    page_df = df.iloc[page * size: page * size + size][[c for c in columns if c in df.columns]].rename(columns=rename)
    return {"data": da._clean_records(page_df), "total": int(total)}
