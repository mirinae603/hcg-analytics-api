"""Executive-summary cards (original frontend contract), the data windows + admin refresh.

`/api/dashboard/all?region=<plant code | name>` returns the exact nested shape the
original AnalyticsDashboardLayout expects (stockAging / kpiStockLevel / returnRate /
daysOnHand / inventoryTurnover). Region that isn't a known plant code => all plants.

`/meta/windows` says which months each source covers and the stock count's date, read from
the data, so no page writes "6 months" or "31 May 2026" in.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from fastapi import APIRouter, Header, HTTPException, Query
from typing import Optional
import numpy as np
import pandas as pd

from app.ai import windows as W
from app.core import data_access as da
from app.core.config import settings

router = APIRouter()


def _resolve_plant(region: Optional[str]) -> Optional[str]:
    return da.resolve_plant(region)


def _count_label() -> Optional[str]:
    """The stock count's date as the cards print it ('31 May 2026'), read from fact_inventory;
    None when there is no count."""
    snap = da.stock_snapshot_date()
    return None if pd.isna(snap) else snap.strftime("%d %b %Y")


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
    # To a year with each part over its own window (internal: the months up to the stock count;
    # billed: the billing months), x 2.0 for both on today's six-month windows.
    avg_itr = da.yearly_cogs(_internal_cogs6, _billed_cogs6) / _inv6 if _inv6 else 0.0

    loc = "All Plants" if not plant else plant
    as_of = _count_label()

    return {
        "stockAging": {
            "fresh": round(fresh, 2), "aging": round(aging, 2),
            "problem": round(problem, 2), "deadStock": round(dead, 2),
            "lastUpdated": as_of,
        },
        "kpiStockLevel": {
            "currentStock": round(stock_qty), "stockValue": round(stock_value, 2),
            "lastMonthRevenue": round(cons_cost, 2), "maxStockValue": round(stock_value * 1.25, 2),
            "monthlyRevenueTarget": round(revenue_proxy, 2), "margin": margin,
            "unit": "Units", "currency": "INR", "label": "Current Stock Value",
            "lowStockThreshold": 1000, "location": loc, "supplier": "—",
            "lastUpdated": as_of,
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
            "location": loc, "lastCalculated": as_of,
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


# ── Data windows (GET /meta/windows) ──────────────────────────────────────────
# Every source covers its own months: the consumption extracts run past the stock count (with a
# month missing), the receipts end before the orders, the legacy billed-sales tables keep
# Dec-25..May-26 on purpose, and the HIS sales and purchase builds publish month by month. The
# pages used to say "6 months", "31 May 2026" and fixed SKU / stock-value figures; they now read
# them here, so a new stock count or a new month changes the words with no code change. A source
# that is not built gives null, never an error (like /export/meta), so one missing table cannot
# blank every label in the app.
WINDOW_ABOUT = {
    "consumption": "Internal goods-issue (SAP consumption extracts), every month received",
    "consumption_to_snapshot": ("The internal issues up to the stock count: what days on hand, turnover, "
                                "non-moving stock and the stock-out list set the counted stock against"),
    "grn": "Goods receipts (SAP GRN): procurement cycle time, purchase value and the GRN export",
    "po": "Purchase orders (SAP PO): the procurement pages",
    "legacy_sales": ("Billed IP and OP sales in the older sales tables (revenue history, billable consumption, "
                     "the billed part of days on hand and turnover), kept to these months on purpose"),
    "sales": "HIS pharmacy sales, published month by month: formulary, spend, revenue and the sales export",
    "purchases": "The purchase build's goods receipts: prices, generic vs branded and out-of-formulary spend",
}


def _window(months) -> Optional[dict]:
    """{from, to, months, gaps} for the months ('YYYY-MM') a source holds; None when it holds none.
    `months` counts the months WITH data and `gaps` names the missing ones inside the span (the
    consumption extracts have no Aug-26), so a label never claims a month that is not there."""
    ms = tuple(sorted(set(months or ())))
    if not ms:
        return None
    return {"from": ms[0], "to": ms[-1], "months": len(ms), "gaps": list(W.missing_months(ms))}


def _source_months(token: str) -> tuple:
    """The months of one SAP transaction table, read exactly as the AI's texts read them
    (app.ai.windows), so a page and the assistant always name the same months."""
    return W.months_of(next(s for s in W.SOURCES if s.token == token))


def _purchase_months() -> tuple:
    """The months the purchase build published (purchase_prices); () while it is not built."""
    from app.api import purchase_insights as PI
    try:
        return tuple(PI._periods(PI.C.PRICES))
    except HTTPException:
        return ()


@lru_cache(maxsize=1)
def _stock_figures() -> dict:
    """The stock count's SKU and value figures, the ones the home screen used to carry as fixed
    numbers. skus_total = materials held at the count or issued in the months up to it (the
    universe the stock and replenishment pages track); skus_in_stock = materials held at the
    count; stock_value = the count's book value at cost. Cleared by POST /admin/refresh-data."""
    out = {"skus_total": None, "skus_in_stock": None, "stock_value": None}
    try:
        sv = da.load("kpi_stock_value")
    except FileNotFoundError:
        return out
    held = set(sv["material"].astype(str))
    try:
        issued = set(da.stock_consumption()["material"].astype(str))
    except FileNotFoundError:
        issued = set()
    out.update(skus_total=len(held | issued), skus_in_stock=len(held),
               stock_value=round(float(sv["stock_value_cost"].sum()), 2))
    return out


def _hospitals() -> Optional[int]:
    """Hospitals with any data: the plants GET /meta/plants lists for the hospital picker, so the
    count and the picker never disagree."""
    from app.api import kpi_generic
    try:
        return sum(1 for p in kpi_generic.meta_plants()["plants"] if p["code"] != "ALL")
    except FileNotFoundError:
        return None


def _built(p: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except FileNotFoundError:
        return None


def _data_as_of() -> Optional[str]:
    """When the newest of the sources was built (UTC, ISO), like the other pages' data_as_of."""
    from app.api import formulary as F, purchase_insights as PI
    built = [_built(da.CURATED / f"{t}.parquet") for t in ("fact_inventory", "fact_consumption", "fact_grn", "fact_po")]
    built += [_built(da.KPI / "sales_monthly.parquet"), F._built_at(), PI._built_at()]
    built = [b for b in built if b]
    return max(built) if built else None


@router.get("/meta/windows")
def meta_windows():
    """The months each source covers and the stock count's date, all read from the data.

    windows: {from, to, months, gaps} or null per source (see WINDOW_ABOUT). consumption is every
    month of internal issues; consumption_to_snapshot only those up to the stock count, the window
    the stock pages and the turnover / days-on-hand rates use. skus_total and stock_value are the
    stock count's own figures, hospitals the plants with data (GET /meta/plants)."""
    snap = da.stock_snapshot_date()
    return {
        "inventory_snapshot": None if pd.isna(snap) else snap.strftime("%Y-%m-%d"),
        "windows": {
            "consumption": _window(_source_months("CONSUMPTION")),
            "consumption_to_snapshot": _window(da.stock_months()),
            "grn": _window(_source_months("RECEIPTS")),
            "po": _window(_source_months("ORDERS")),
            "legacy_sales": _window(da.legacy_sales_months()),
            "sales": _window(W.report_months()),
            "purchases": _window(_purchase_months()),
        },
        **_stock_figures(),
        "hospitals": _hospitals(),
        "about": WINDOW_ABOUT,
        "data_as_of": _data_as_of(),
    }


def clear_caches() -> None:
    _stock_figures.cache_clear()


@router.post("/admin/refresh-data")
def refresh_data(x_admin_token: Optional[str] = Header(None)):
    if x_admin_token != settings.ADMIN_REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid admin token")
    da.refresh_cache()
    clear_caches()
    # Also drop the memoized portfolio-overview / summary results so they recompute
    # against the refreshed data (otherwise they'd serve the previous snapshot).
    try:
        from app.api import legacy_kpi, kpi_generic, formulary, spend
        legacy_kpi.clear_result_caches()
        kpi_generic._portfolio_summary_cached.cache_clear()
        kpi_generic.clear_kpi_caches()
        # which plants have stock / consumption / receipts: a new stock count can change them,
        # and both GET /meta/plants and the hospitals count of GET /meta/windows read it
        kpi_generic._plant_domains.cache_clear()
        formulary.clear_caches()
        spend.clear_caches()
    except Exception:
        pass
    # The AI's prompt text (schema, grain, semantics context, resolve() answers) is cached
    # for the process too. Clear it only if the AI side is already loaded: importing it here
    # would open the warehouse on a plain data refresh, and a module never imported has
    # nothing cached.
    wh = sys.modules.get("app.ai.warehouse")
    if wh is not None:
        try:
            wh.reset_caches()
        except Exception:
            pass
    # Refill the executive overview's caches (and each section's default view) in the background,
    # so the first visitor after a refresh does not wait for them. It never blocks this answer,
    # and does nothing outside a serving process, e.g. under the tests (app/api/warmup.py).
    from app.api import warmup
    warmup.start("refresh")
    return {"status": "cache cleared"}
