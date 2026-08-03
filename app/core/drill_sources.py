"""Richer-grain sources for the hover drill-down.

Several KPI aggregates are pre-aggregated PAST the grain a drill-down needs. Asking
them "what are the top 10 materials inside this bar" cannot work, because they have
no material column at all:

    kpi_aging_distribution  plant x material_group x aging_bucket   (no material)
    kpi_vendor_volume       plant x vendor                          (no material)
    kpi_purchase_value      plant x vendor x PO-category x month    (no material)
    kpi_purchase_by_location / kpi_fill_rate   one row per plant    (no material)
    kpi_cycle_time          plant x month                           (no material/vendor)
    kpi_consumption_by_department  plant x cost centre x month      (no material)

The grain is not missing from the data, only from the aggregate — every one of those
tables was built by transforms.py from a fact table that DOES carry material (and
vendor, and cost centre). So instead of rejecting the request, the drill endpoint
resolves to a frame that actually has the grain being asked for.

Each builder below rebuilds one fact table ONCE into a slim cached frame, using the
SAME derivations transforms.py used — the same aging bins, the same
`major_group -> Uncategorized` fill, the same lead-time clipping — so a drill that
falls back to one of these reproduces its parent bar to the rupee. tests/
test_drilldown.py asserts that equality directly against the pre-aggregated tables.

Two naming rules make the fallbacks substitutable for their parents:

  * MEASURE ALIASES. Each frame carries the parent's own measure name as a column
    (`vendor_value`, `monthly_purchase_value`, `closing_stock_value`, …) even when it
    is the same number under another name, so a caller never has to know it was
    re-routed.
  * CATEGORY COLUMNS. `category` always means whatever the PARENT table means by it
    — the PO spend taxonomy on the procurement frame (mirroring kpi_purchase_value),
    the derived material category everywhere else. The derived material category is
    additionally always available as `material_category`, and the drill endpoint
    filters on the column named in DERIVED_CATEGORY_COL rather than on `category`,
    so the global Category selector can never be applied to the wrong vocabulary.

Memory: every frame is an aggregate a fraction of its fact's size (fact_po 250k rows
-> 157k, fact_grn 256k -> 151k, fact_inventory 98k -> 69k, fact_consumption 128k ->
83k) with string columns held as pandas categoricals — ~45 MB resident for all four,
and each is built lazily on the first drill that needs it, never at boot.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.core import data_access as da

# Identical to app/etl/transforms.py — the ladder kpi_aging_distribution was cut with.
AGING_BINS = [-1, 30, 90, 180, 365, np.inf]
AGING_LABELS = ["0-30", "31-90", "91-180", "181-365", "365+"]

UNMAPPED_GROUP = "Unmapped"


def _as_category(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].dtype.name != "category":
            df[c] = df[c].astype("category")
    return df


@lru_cache(maxsize=1)
def _material_group_map() -> dict:
    dm = da.load("dim_material")
    return {str(m): (str(g) if pd.notna(g) and str(g) not in ("nan", "") else UNMAPPED_GROUP)
            for m, g in zip(dm["material"], dm["material_group"])}


@lru_cache(maxsize=1)
def material_desc_map() -> dict:
    dm = da.load("dim_material")
    return {str(m): str(d) for m, d in zip(dm["material"], dm["material_desc"])
            if pd.notna(d) and str(d) not in ("nan", "")}


@lru_cache(maxsize=1)
def department_name_map() -> dict:
    dc = da.load("dim_costcenter")
    return {str(c): str(n) for c, n in zip(dc["cost_ctr"], dc["department_name"])
            if pd.notna(n) and str(n) not in ("nan", "")}


@lru_cache(maxsize=1)
def plant_name_map() -> dict:
    dp = da.load("dim_plant")
    return {str(c): str(n) for c, n in zip(dp["plant"], dp["plant_name"])}


def _group_of(materials: pd.Series) -> pd.Series:
    return materials.astype(str).map(_material_group_map()).fillna(UNMAPPED_GROUP)


# ── fact_po: the grain behind every procurement aggregate ─────────────────────
@lru_cache(maxsize=1)
def po_grain() -> pd.DataFrame:
    """plant x vendor x PO-category x material x month.

    Parent of kpi_purchase_value (B1), kpi_monthly_purchase_value (B2),
    kpi_procurement_variance (B3), kpi_vendor_volume (B4),
    kpi_purchase_by_location (B7) and kpi_fill_rate (E4) — all six are
    `fact_po.groupby(...)` over `total_value_wo_tax` / `po_qty` / `open_qty`, so
    grouping this frame the same way reproduces any of them exactly.
    """
    po = da.load("fact_po")               # `category` = derived material category
    po = po.assign(
        po_category=po["major_group"].replace({"nan": np.nan}).fillna("Uncategorized"),
        material_group=_group_of(po["material"]),
    )
    g = (po.groupby(["plant", "vendor_name", "po_category", "category", "material",
                     "material_group", "year", "month"], dropna=False, observed=True)
           .agg(purchase_value=("total_value_wo_tax", "sum"),
                purchase_qty=("po_qty", "sum"),
                open_qty=("open_qty", "sum"),
                po_lines=("po_no", "count"))
           .reset_index()
           .rename(columns={"category": "material_category"}))
    # `category` on kpi_purchase_value IS the PO spend taxonomy — mirror it, so
    # dim=category means the same thing whichever source answers the request.
    g["category"] = g["po_category"]
    g["vendor_value"] = g["purchase_value"]            # kpi_vendor_volume's measure
    g["monthly_purchase_value"] = g["purchase_value"]  # kpi_monthly_purchase_value's
    g["ordered_qty"] = g["purchase_qty"]               # kpi_fill_rate's
    g["fill_rate_pct"] = np.where(g["ordered_qty"] > 0,
                                  (1 - g["open_qty"] / g["ordered_qty"]) * 100, np.nan)
    return _as_category(g, ["plant", "vendor_name", "po_category", "category",
                            "material_category", "material_group", "month", "material"])


# ── fact_grn: the grain behind the lead-time / cycle-time aggregates ──────────
@lru_cache(maxsize=1)
def grn_grain() -> pd.DataFrame:
    """plant x vendor x material x month, carrying lead-time SUMS and COUNTS.

    Parent of kpi_cycle_time (E2) and kpi_vendor_lead_time (E3). Those are means, so
    the frame keeps `tat_sum` / `tat_n` rather than a pre-divided average: a drill can
    then aggregate them as sum(tat)/count(tat), which is the exact weighted mean
    rather than an average-of-averages. Applies transforms.py's own clipping first
    (negative TAT = GR before PO = a data error, > 365 days = an implausible outlier).
    """
    grn = da.load("fact_grn")
    grn = grn.assign(material_group=_group_of(grn["material"]))
    for c in ("po_to_gr_tat", "pr_to_gr_tat"):
        s = pd.to_numeric(grn[c], errors="coerce")
        grn[c] = s.mask((s < 0) | (s > 365))
    g = (grn.groupby(["plant", "vendor_name", "material", "material_group", "category",
                      "year", "month"], dropna=False, observed=True)
            .agg(grn_value=("total_amount_wo_tax", "sum"),
                 grn_qty=("gr_qty", "sum"),
                 gr_lines=("gr_no", "count"),
                 po_tat_sum=("po_to_gr_tat", "sum"), po_tat_n=("po_to_gr_tat", "count"),
                 pr_tat_sum=("pr_to_gr_tat", "sum"), pr_tat_n=("pr_to_gr_tat", "count"))
            .reset_index()
            .rename(columns={"category": "material_category"}))
    g["category"] = g["material_category"]
    with np.errstate(invalid="ignore", divide="ignore"):
        g["avg_po_to_gr_tat"] = np.where(g["po_tat_n"] > 0, g["po_tat_sum"] / g["po_tat_n"], np.nan)
        g["avg_pr_to_gr_tat"] = np.where(g["pr_tat_n"] > 0, g["pr_tat_sum"] / g["pr_tat_n"], np.nan)
    g["avg_lead_time_days"] = g["avg_po_to_gr_tat"]
    g["median_lead_time_days"] = g["avg_po_to_gr_tat"]   # weighted-mean approximation
    return _as_category(g, ["plant", "vendor_name", "material_group", "category",
                            "material_category", "month", "material"])


# ── fact_inventory: the grain behind the pre-aggregated aging distribution ────
@lru_cache(maxsize=1)
def inventory_grain() -> pd.DataFrame:
    """plant x material x material_group x aging_bucket.

    Parent of kpi_aging_distribution (A7) — which is exactly this frame collapsed
    onto (plant, material_group, aging_bucket), i.e. with the material column thrown
    away. Cut with transforms.py's own AGING_BINS so the buckets are the same
    buckets, not a lookalike ladder. `sku_count` is 1 per row, so summing it back up
    reproduces A7's `nunique(material)` per cell exactly.
    """
    inv = da.load("fact_inventory")       # `category` derived from material_type
    inv = inv.assign(
        aging_bucket=pd.cut(inv["aging_days"], AGING_BINS, labels=AGING_LABELS).astype(str),
        _age_qty=pd.to_numeric(inv["aging_days"], errors="coerce").fillna(0.0)
        * pd.to_numeric(inv["qty"], errors="coerce").fillna(0.0).clip(lower=0.0001),
        _qty_w=pd.to_numeric(inv["qty"], errors="coerce").fillna(0.0).clip(lower=0.0001),
    )
    g = (inv.groupby(["plant", "material", "material_group", "category", "aging_bucket"],
                     dropna=False, observed=True)
            .agg(stock_value=("total_cost", "sum"),
                 stock_qty=("qty", "sum"),
                 stock_value_mrp=("total_mrp", "sum"),
                 batches=("batch", "nunique"),
                 _age_qty=("_age_qty", "sum"), _qty_w=("_qty_w", "sum"))
            .reset_index())
    g["aging_days"] = np.where(g["_qty_w"] > 0, g["_age_qty"] / g["_qty_w"], 0.0)
    g = g.drop(columns=["_age_qty", "_qty_w"])
    g["material_category"] = g["category"]
    g["sku_count"] = 1
    g["stock_value_cost"] = g["stock_value"]        # kpi_stock_value's measure
    g["closing_stock_value"] = g["stock_value"]     # kpi_inventory_aging / health / risk
    g["total_cost"] = g["stock_value"]              # kpi_near_expiry's
    g["closing_stock_quantity"] = g["stock_qty"]
    g["total_mrp"] = g["stock_value_mrp"]
    return _as_category(g, ["plant", "material_group", "category", "material_category",
                            "aging_bucket", "material"])


# ── fact_consumption: the grain behind the department aggregate ───────────────
@lru_cache(maxsize=1)
def consumption_grain() -> pd.DataFrame:
    """plant x cost centre x material x month.

    Parent of kpi_units_consumed (C8) and kpi_consumption_by_department (C9) — C9 is
    this frame with the material column dropped, which is why a department bar could
    never say what was consumed inside it.
    """
    con = da.load("fact_consumption")     # `category` derived from material
    con = con.assign(material_group=_group_of(con["material"]))
    g = (con.groupby(["plant", "cost_ctr", "material", "material_group", "category",
                      "year", "month"], dropna=False, observed=True)
            .agg(consumption_qty=("qty", "sum"),
                 consumption_cost=("amount_lc", "sum"),
                 movements=("mvt", "count"))
            .reset_index()
            .rename(columns={"category": "material_category"}))
    g["category"] = g["material_category"]
    g["total_units"] = g["consumption_qty"]         # kpi_units_consumed's measure
    return _as_category(g, ["plant", "cost_ctr", "material_group", "category",
                            "material_category", "month", "material"])


# ── kpi_doh + kpi_stock_value: the grain behind the coverage histogram ────────
# The cover ladder the Days-on-Hand page draws. Copied from legacy_kpi.DOH_BANDS
# rather than re-invented, and asserted equal to it in tests/test_drilldown.py, so
# the drill can never band a SKU differently from the bar it was launched from.
DOH_BANDS = [("critical", 0, 15), ("low", 15, 30), ("healthy", 30, 90),
             ("ample", 90, 180), ("excess", 180, 365), ("overstock", 365, np.inf)]
DOH_NONMOVING = "nonmoving"


@lru_cache(maxsize=1)
def doh_grain() -> pd.DataFrame:
    """plant x material carrying `doh_band` — the cover band, as a real column.

    GET /kpi/days-on-hand/insights COMPUTES its six coverage bands from `doh_days`
    and values them off kpi_stock_value, which it merges in; neither the band nor the
    rupee value it is measured in exists on kpi_doh, so `dim=doh_band` had nothing to
    group by and `measure=stock_value_cost` no column to rank on.

    This is that endpoint's own first four lines — the same merge, the same
    `fillna(0)`, the same `>= lo & < hi` bands, the same `doh_days.isna()` definition
    of "no movement" — materialised once and cached, so summing `stock_value_cost`
    per band here reproduces the histogram bar for bar and `sku_count` reproduces its
    SKU counts.
    """
    doh = da.load("kpi_doh")
    sv = da.load("kpi_stock_value")[["plant", "material", "stock_value_cost", "stock_value_mrp"]]
    g = doh.merge(sv, on=["plant", "material"], how="left")
    for c in ("stock_value_cost", "stock_value_mrp"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    days = pd.to_numeric(g["doh_days"], errors="coerce")
    # A SKU with no consumption has no cover at all — it is not "0 days", which would
    # sweep 21k idle SKUs into the critical band and treble the stockout figure.
    band = pd.Series(DOH_NONMOVING, index=g.index, dtype=object)
    for key, lo, hi in DOH_BANDS:
        band = band.mask(days.notna() & (days >= lo) & (days < hi), key)
    g["doh_band"] = band.astype(str)

    g["material_category"] = g["category"]
    g["sku_count"] = 1                      # one row per plant x material — sums to the count
    g["stock_value"] = g["stock_value_cost"]
    return _as_category(g, ["plant", "material_group", "category", "material_category",
                            "doh_band", "material"])


# ── kpi_near_expiry, expired lines only: the numerator of the wastage ratio ───
@lru_cache(maxsize=1)
def expired_grain() -> pd.DataFrame:
    """plant x material x batch, restricted to the Expired bucket.

    Wastage % is expired_value / stock_value — a ratio of two tables, which is why
    it has no registry KPI and why ranking SKUs BY it would be nonsense. The rupee
    NUMERATOR is perfectly additive though, and "which items expired at this
    hospital" is exactly the next question in front of a plant bar. So the drill
    ranks this frame by `expired_value`, and the shares it reports are shares of that
    plant's expired value — never of the percentage.

    Cut from da.load('kpi_near_expiry') (which carries the days_to_expiry == 0
    boundary fix) with legacy_kpi._wastage_frame's own `expiry_bucket == "Expired"`
    predicate, so the per-plant totals here equal that page's `expired_value` exactly.
    """
    ne = da.load("kpi_near_expiry")
    g = ne[ne["expiry_bucket"].astype(str) == "Expired"].copy()
    g["expired_value"] = pd.to_numeric(g["total_cost"], errors="coerce").fillna(0.0)
    g["expired_qty"] = pd.to_numeric(g["qty"], errors="coerce").fillna(0.0)
    g["material_category"] = g["category"]
    return _as_category(g, ["plant", "material_group", "category", "material_category",
                            "expiry_bucket", "material"])


# ── replenishment queue with its priority bands ───────────────────────────────
def replen_priority(plant: Optional[str] = None) -> pd.DataFrame:
    """The reorder queue carrying `priority_band` / `priority_label` / `status`.

    Those three columns are COMPUTED (cover vs demand vs stock), not stored on the
    replenishment parquet — which is why dim=priority_band had nothing to group by.
    Delegates to legacy_kpi._replen_frame, the exact function GET
    /forecast/reorder-priority uses, so a band drill-down and the band bar it was
    launched from can never disagree.
    """
    from app.api import legacy_kpi
    return legacy_kpi._replen_frame(plant)


# name -> builder(plant) . Everything else resolves through da.load(<table>).
BUILDERS: dict[str, Callable[[Optional[str]], pd.DataFrame]] = {
    "drill_po_grain": lambda _p: po_grain(),
    "drill_grn_grain": lambda _p: grn_grain(),
    "drill_inventory_grain": lambda _p: inventory_grain(),
    "drill_consumption_grain": lambda _p: consumption_grain(),
    "drill_doh_grain": lambda _p: doh_grain(),
    "drill_expired_grain": lambda _p: expired_grain(),
    "drill_replen_priority": replen_priority,
}

# Which column on each source holds the DERIVED material category, i.e. the one the
# global Category selector is allowed to filter. Defaults to `category`; the
# procurement frames override it because their `category` is the PO spend taxonomy.
DERIVED_CATEGORY_COL: dict[str, str] = {
    "drill_po_grain": "material_category",
    "kpi_purchase_value": "material_category",   # has none — filtering stays a no-op
}


def load_source(name: str, plant: Optional[str]) -> pd.DataFrame:
    b = BUILDERS.get(name)
    return b(plant) if b else da.load(name)


def clear_caches() -> None:
    for fn in (po_grain, grn_grain, inventory_grain, consumption_grain,
               doh_grain, expired_grain,
               _material_group_map, material_desc_map, department_name_map, plant_name_map):
        fn.cache_clear()
