"""In-memory parquet access layer + server-side table filtering/pagination.

Replaces the old pyodbc/Azure-SQL data layer. KPI aggregate parquet (produced by
the ETL) are memoized on first read. `paginate` is a pandas port of the old
`ui_table_filter_controls.build_filter_clause` so every `*-table` endpoint keeps the
identical `{data,total}` contract and the `filter_field_i/operator_i/value_i` protocol.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.config import settings

KPI = Path(settings.KPI_DIR)
CURATED = Path(settings.CURATED_DIR)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# The big fact tables (fact_grn/fact_po ≈ 84 MB each) are NOT held resident: keeping
# them all in the lru_cache pushes RSS past the 512 MB free-tier limit → OOM. They load
# fresh per request and are freed right after (the heavy endpoints that scan them are
# result-cached, so this reload happens rarely). Small KPI aggregates stay cached.
_BIG_TABLES = {"fact_grn", "fact_po", "fact_inventory", "fact_consumption", "forecast_sales"}


# ── Post-load corrections for defects baked into committed parquet ────────────
# kpi_near_expiry.expiry_bucket was written by an ETL `pd.cut(..., [-9999, 0, 30,
# 90, 180])` whose right-closed first bin (-9999, 0] swept days_to_expiry == 0
# into "Expired". A batch whose expiry date IS the snapshot date has not expired —
# it is dispensable through today and is the most urgent RECOVERABLE line, so it
# belongs in "0-30d" (a bucket whose label was otherwise a lie: it held only 1-30d).
# legacy_kpi.py's expiry ladder and the risk matrix's eb() both already treat
# days_to_expiry < 0 as "Expired"; this realigns the baked column with them for
# every consumer at once, without regenerating (and re-committing) the parquet.
# transforms.py is fixed to match, so this becomes a no-op once the ETL next runs.
_NEAR_EXPIRY_BINS = [-10**12, -1, 30, 90, 180]
_NEAR_EXPIRY_LABELS = ["Expired", "0-30d", "31-90d", "91-180d"]


# ── Derived `category` dimension ──────────────────────────────────────────────
# HCG's material taxonomy lives ONLY in dim_material.material_type (SAP material
# types like "ZOC-Medical Onco Drugs"); no KPI aggregate carries it. Rather than
# regenerate + re-commit every parquet, we derive a `category` column at the single
# _read_parquet choke point, exactly as the near-expiry bucket fix above does — so
# every consumer (charts, tables, insights, the AI analyst) gets it for free.
#
# The buckets are chosen for an ONCOLOGY hospital chain: the onco / non-onco drug
# split is the single most decision-relevant cut, so it is NEVER collapsed into one
# "Pharma". Verified shares of the Rs 60.47 Cr total stock value:
#   Onco Drugs 41.67% | Consumables 36.44% | Other Drugs 17.03% | Non-Medical 3.49%
#   | Lab 1.38% | Unclassified 0.00%
# ~6,851 dim_material rows carry a null material_type, but they hold ZERO stock
# value — so the filter is complete on value even though it is not on raw row count.
# They (and any material absent from dim_material entirely) land in "Unclassified":
# clearly labelled, never silently dropped and never silently folded into a real
# bucket.
CATEGORY_UNCLASSIFIED = "Unclassified"

# SAP material-type code (the part before the first "-") -> reporting bucket.
CATEGORY_CODE_MAP = {
    "ZOC": "Onco Drugs",
    "ZNOC": "Other Drugs",
    "ZMC": "Consumables",
    "ZLR": "Lab",
    "ZLCL": "Lab",
    "ZLCO": "Lab",
    "ZNMC": "Non-Medical",
    "ZNMA": "Non-Medical",
    "ZMA": "Non-Medical",
}

# Display order the frontend should render (excluding the implicit "All" default).
CATEGORIES = ["Onco Drugs", "Other Drugs", "Consumables", "Lab", "Non-Medical",
              CATEGORY_UNCLASSIFIED]

# Every string a caller may legitimately pass for a bucket, lowercased.
_CATEGORY_ALIASES = {c.lower(): c for c in CATEGORIES}
_CATEGORY_ALIASES.update({k.lower(): v for k, v in CATEGORY_CODE_MAP.items()})


def category_of_material_type(mt) -> str:
    """Map one raw dim_material.material_type value to its reporting bucket."""
    s = "" if mt is None else str(mt).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return CATEGORY_UNCLASSIFIED
    return CATEGORY_CODE_MAP.get(s.split("-", 1)[0].strip().upper(), CATEGORY_UNCLASSIFIED)


@lru_cache(maxsize=1)
def _material_category_map() -> dict:
    """material -> category, built ONCE from dim_material.

    Read straight off the parquet rather than through load()/_read_parquet: this is
    called from inside _normalize, so going back through the normal read path would
    recurse. dim_material is ~25k rows, so the one-off build is trivial and the
    lru_cache means the per-table hook below is a single vectorised .map().
    """
    p = CURATED / "dim_material.parquet"
    if not p.exists():
        return {}
    dm = pd.read_parquet(p, columns=["material", "material_type"])
    return {str(m): category_of_material_type(t)
            for m, t in zip(dm["material"], dm["material_type"])}


def _material_key(df: pd.DataFrame) -> Optional[str]:
    for c in ("material", "material_id"):
        if c in df.columns:
            return c
    return None


def _attach_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `category` column to any frame that carries a material dimension.

    Order of preference:
      1. an existing `material_type` column (fact_inventory / dim_material carry
         their own — use it directly, no join needed);
      2. a material / material_id column, mapped through dim_material.
    Frames with neither (kpi_purchase_value, kpi_vendor_volume, kpi_cycle_time,
    kpi_aging_distribution, …) are left completely untouched — no column, no row
    change — so a category cut can never be silently faked on a table that has no
    material grain. A pre-existing `category` column (kpi_purchase_value's PO
    category) is likewise never overwritten.
    """
    if "category" in df.columns:
        return df

    # This hook runs on EVERY table load, including the big fact tables that are
    # deliberately not held resident and so reload per request — it has to stay cheap.
    # Both branches below are a single vectorised .map() against a prebuilt dict; the
    # material_type branch builds its lookup from that column's ~10 DISTINCT values
    # rather than calling the classifier once per row (which cost ~50 ms on
    # fact_inventory's 97k rows for no reason). astype() is skipped when the key
    # column is already a string dtype, which it is in every committed parquet.
    def _as_str(s):
        return s if s.dtype == "str" or s.dtype == object else s.astype("str")

    if "material_type" in df.columns:
        uniq = df["material_type"].dropna().unique()
        lut = {u: category_of_material_type(u) for u in uniq}
        df["category"] = df["material_type"].map(lut).fillna(CATEGORY_UNCLASSIFIED).astype("str")
        return df

    key = _material_key(df)
    if key is None:
        return df
    cmap = _material_category_map()
    if not cmap:
        return df
    df["category"] = _as_str(df[key]).map(cmap).fillna(CATEGORY_UNCLASSIFIED).astype("str")
    return df


def _normalize(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if table == "kpi_near_expiry" and {"days_to_expiry", "expiry_bucket"} <= set(df.columns):
        dtype = df["expiry_bucket"].dtype
        df["expiry_bucket"] = pd.cut(df["days_to_expiry"], _NEAR_EXPIRY_BINS,
                                     labels=_NEAR_EXPIRY_LABELS,
                                     right=True).astype(str).astype(dtype)
    return _attach_category(df)


def _read_parquet(table: str) -> pd.DataFrame:
    for base in (KPI, CURATED):
        p = base / f"{table}.parquet"
        if p.exists():
            return _normalize(table, pd.read_parquet(p))
    raise FileNotFoundError(f"parquet not found: {table}")


@lru_cache(maxsize=128)
def _load_cached(table: str) -> pd.DataFrame:
    return _read_parquet(table)


def load(table: str) -> pd.DataFrame:
    """Load a KPI aggregate (or curated) parquet. Small tables are cached; the big fact
    tables load fresh each call so they don't stay resident (512 MB free-tier budget)."""
    if table in _BIG_TABLES:
        return _read_parquet(table)
    return _load_cached(table)


def refresh_cache() -> None:
    _load_cached.cache_clear()
    _name_to_code.cache_clear()
    _material_category_map.cache_clear()
    sales_material_margin.cache_clear()


@lru_cache(maxsize=1)
def sales_material_margin() -> pd.DataFrame:
    """One row per BILLED MATERIAL, with margin — the frame behind the Revenue &
    Margin detail table.

    sales_by_material.parquet cannot be served as-is, for three separate reasons:

    1. It stores the SAME product under two ids — "218766" and "218766.0". 7,320 of
       its 21,960 rows carry the ".0" suffix, so a raw read reports 21,960 materials
       where there are 15,171, and splits a KEYTRUDA-class product across two rows
       that each hold half its revenue. legacy_kpi.revenue_insights already repairs
       this for its own top_items/materials count; this is the same repair, done once
       and shared, so the table underneath that page cannot disagree with it.
    2. It carries no `margin` or `margin_pct` column at all — both are derived here.
       margin_pct is computed PER ROW from that row's own revenue, never summed:
       averaging a rate across materials is meaningless and summing it is worse.
    3. Its `category` (attached by _attach_category at read time) is wrong on every
       ".0" row, because "218766.0" misses dim_material's "218766". Categorising
       AFTER the normalise lifts Onco Drugs from 708 to 733 materials and drops
       Unclassified from 5,507 to 4,214 — without it a Category=Onco Drugs cut
       understates billed revenue by ~4.4% (297.5 Cr served vs 311.1 Cr true).

    The statement order below is load-bearing:
      normalise the id  ->  sort by the trimmed group DESCENDING so a non-blank group
      sorts ahead of a blank one  ->  groupby(material).agg(group="first").
    Without that sort, agg("first") is free to pick the blank half of a split pair and
    the material lands in "Uncategorised" despite the extract knowing its group. Only
    then is the category attached, and only then are margin and margin_pct derived.

    NOTE: this extract has NO plant column. Anything built on it is network-wide; a
    Plant/Hospital filter over it is genuinely inert, and callers must say so rather
    than imply a hospital cut that the source cannot support.
    """
    # Imported lazily: app.api.legacy_kpi imports this module at its top, so a
    # module-level import here would be circular. By the time this function is first
    # CALLED (from a request handler) both modules are fully loaded.
    from app.api.legacy_kpi import _clean_group

    p = KPI / "sales_by_material.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["material", "desc", "group", "group_label", "category",
                                     "revenue", "cost", "margin", "margin_pct", "qty", "lines"])
    df = pd.read_parquet(p)
    df["material"] = (df["material"].astype(str)
                      .str.replace(r"\.0$", "", regex=True).str.strip())
    df["_g"] = df["group"].astype(str).str.strip()
    df = df.sort_values("_g", ascending=False, kind="mergesort")

    agg = {"revenue": ("revenue", "sum"), "cost": ("cost", "sum"), "qty": ("qty", "sum")}
    if "lines" in df.columns:
        agg["lines"] = ("lines", "sum")
    if "desc" in df.columns:
        agg["desc"] = ("desc", "first")
    m = df.groupby("material", as_index=False).agg(group=("_g", "first"), **agg)

    m = _attach_category(m)          # AFTER the normalise — see (3) above
    # RAW group kept untouched ("M065-INJECTIONS"); the pretty form lives beside it.
    # Anything sent back to an API must use the raw key, never the label.
    m["group_label"] = m["group"].map(_clean_group)
    m["margin"] = m["revenue"] - m["cost"]
    m["margin_pct"] = np.where(m["revenue"] > 0, m["margin"] / m["revenue"] * 100.0, 0.0)
    return m.sort_values("revenue", ascending=False).reset_index(drop=True)


def nonmoving_scoped(scope: str) -> pd.DataFrame:
    """Recomputes A9 (kpi_non_moving) with a different "moving" material definition.

    Shared by legacy_kpi.py's /kpi/non-moving-inventory/insights and kpi_generic.py's
    /kpi/non-moving-inventory/table so both surfaces agree under a Scope filter.
    Reproduces transforms.py's A9 formula verbatim (isin -> aging>180 OR-filter ->
    reason label) against kpi_inventory_aging, the pre-classification universe A9 is
    itself derived from, swapping only which materials count as "moving": today's A9
    (scope "nonbillable", never reaches this function) uses fact_consumption alone.
    "billable" uses sales_by_material's material set instead (patient-billed, IP+OP);
    "both" unions the two — the exact gap the DOH RCA identified: injectable/oncology
    materials dispensed via billing but never internally issued get wrongly classified
    "non-moving" today.
    """
    a2 = load("kpi_inventory_aging")  # already carries its own last_sale_date (see transforms.py A2)
    cons = load("fact_consumption")
    moving = set()
    if scope in ("billable", "both"):
        sm_path = KPI / "sales_by_material.parquet"
        if sm_path.exists():
            moving |= set(pd.read_parquet(sm_path)["material"].astype(str).unique())
    if scope == "both":
        moving |= set(cons["material"].astype(str).unique())

    a9 = a2.copy()
    a9["consumed_in_window"] = a9["material"].astype(str).isin(moving)
    a9 = a9[(~a9["consumed_in_window"]) | (a9["aging_days"] > 180)].copy()
    a9["reason"] = np.where(~a9["consumed_in_window"], "No consumption in 6mo", "Aging > 180d")
    # `category` carries over from kpi_inventory_aging (already attached at load()
    # time via _attach_category) so filter_category still works downstream — a plain
    # column-select without it would silently no-op any Category filter combined with
    # Scope=billable|both, since _attach_category only runs inside load()/_read_parquet,
    # never on an in-memory frame built after the fact.
    return a9[["plant", "material", "material_desc", "material_group", "category",
               "closing_stock_quantity", "closing_stock_value", "aging_days",
               "last_sale_date", "reason"]]


# sales_by_material retains no per-row date to derive a dynamic span from (unlike
# fact_consumption's own days_span, computed live from real posting dates). This is
# the fixed Dec 2025-May 2026 window ingest_sales.py's own KEEP set already bakes in
# ("match the rest of the data") -- 31+31+28+31+30+31 days, not a separate guess.
_SALES_WINDOW_DAYS = 182


def doh_scoped(scope: str) -> pd.DataFrame:
    """Recomputes A3 (kpi_doh) folding in the patient-billed consumption rate.

    "nonbillable" (today's default, never reaches this function) leaves
    avg_daily_consumption/doh_days exactly as ETL wrote them, fact_consumption-only.
    "billable" replaces avg_daily_consumption with the billed qty/day rate from
    sales_by_material; "both" SUMS the internal and billed daily-quantity rates
    before dividing stock_qty by the total -- this is deliberately a units/day figure
    both sides, never mixed with revenue/cost, so the two rates stay unit-compatible.
    This is the DOH RCA's own proposed fix, offered as an optional lens rather than a
    silent change to the shipped default.

    Caveat surfaced in the UI, not hidden here: sales_by_material carries no plant
    dimension, so the billed rate is ONE network-wide qty/day per material applied to
    every plant that stocks it, not a true per-hospital rate.
    """
    doh = load("kpi_doh").copy()
    billed_daily = pd.Series(dtype=float)
    if scope in ("billable", "both"):
        sm_path = KPI / "sales_by_material.parquet"
        if sm_path.exists():
            sm = pd.read_parquet(sm_path, columns=["material", "qty"])
            billed_daily = sm.groupby("material")["qty"].sum() / _SALES_WINDOW_DAYS
    doh["material"] = doh["material"].astype(str)
    doh["billed_daily"] = doh["material"].map(billed_daily).fillna(0.0)
    if scope == "billable":
        doh["avg_daily_consumption"] = doh["billed_daily"]
    elif scope == "both":
        doh["avg_daily_consumption"] = doh["avg_daily_consumption"] + doh["billed_daily"]
    doh["doh_days"] = np.where(doh["avg_daily_consumption"] > 0,
                               doh["stock_qty"] / doh["avg_daily_consumption"], np.nan)
    return doh.drop(columns=["billed_daily"])


def doh_value_scoped(plant: Optional[str], category: Optional[str]) -> dict:
    """Days Inventory Outstanding, client's own formula: Closing Inventory VALUE /
    (avg daily VALUE of goods sold+consumed). This is a DIFFERENT metric from
    doh_scoped's doh_days (unit quantity, per-SKU MEDIAN) -- a single, portfolio-
    level, VALUE-weighted ratio: SUM(closing_stock_value) / (SUM(internal cost)/181
    + SUM(billed cost)/182). Verified against real data: ~29.3 days at All Plants,
    vs. the unit-median DOH's 16.57 -- both correct, answering different questions
    (typical SKU's cover vs. rupee-weighted portfolio cover, which skews toward
    high-value slow movers exactly the way the mean-vs-median doh_days gap already
    does). Ships alongside doh_days, not replacing it: bands/reorder/overstock
    still need real per-SKU quantities, which a single aggregate ratio can't give.

    PLANT CAVEAT (real, load-bearing, discovered live -- not theoretical): billed
    cost (sales_by_material) carries no plant dimension, same limitation doh_scoped
    already has. doh_scoped absorbs that by blending the network-wide billed rate
    into each SKU's own row before taking a MEDIAN across thousands of rows, which
    dilutes any one mismatch. THIS function has no such buffer -- it is a single
    SUM/SUM ratio, so blending the network's entire billed total into one hospital's
    denominator collapsed the number to 2-4 days (verified: HC05 -> 4.4d, HO01 ->
    2.5d test values, both absurd). So: billed cost is folded in ONLY at the true
    All-Plants view; a single-plant selection returns an honest INTERNAL-ONLY days
    figure instead (real number, narrower scope) rather than a blended, meaningless
    one. `billed_applicable` in the return tells the caller which case fired, so the
    UI can disclose it instead of leaving the scope change silent.

    CATEGORY CAVEAT: sales_by_material carries its own `category` bucket (attached
    at load() time, NOT the raw material_group column) -- filtering billed cost by
    `group`/material_group instead of `category` silently returns nothing for every
    real category value here (confirmed live: an M0xx- style filter zeroed both
    kpi_stock_value AND fact_consumption, since neither carries that column at all;
    they carry `category`). Category filtering works at ANY plant scope, unlike
    Plant -- Onco Drugs is the proof this matters: ~0 internal consumption but
    Rs 197 Cr of billed cost, so the category filter is worthless here unless billed
    is correctly included (verified: 23.24 days once wired correctly, not the
    near-infinite number a consumption-only view would show for a near-zero
    denominator).
    """
    p = resolve_plant(plant)
    cat = resolve_category(category)

    sv = filter_category(filter_plant(load("kpi_stock_value"), p), cat)
    closing_value = float(sv["stock_value_cost"].sum())

    cons_all = load("fact_consumption")
    days_span = max((cons_all["posting_date"].max() - cons_all["posting_date"].min()).days, 1)
    cons_f = filter_category(filter_plant(cons_all, p), cat)
    internal_value = float(cons_f["amount_lc"].sum())
    internal_daily = internal_value / days_span

    billed_applicable = p is None
    billed_value = 0.0
    if billed_applicable:
        sm = load("sales_by_material")
        if cat:
            sm = sm[sm["category"].astype(str) == cat]
        billed_value = float(sm["cost"].sum())
    billed_daily = billed_value / _SALES_WINDOW_DAYS

    avg_daily_value = internal_daily + billed_daily
    days = (closing_value / avg_daily_value) if avg_daily_value > 0 else None
    return {
        "closing_inventory_value": closing_value,
        "internal_consumption_value": internal_value,
        "billed_cost_value": billed_value,
        "billed_applicable": billed_applicable,
        "days_span": days_span,
        "avg_daily_value": avg_daily_value,
        "days_inventory_value": days,
    }


def itr_scoped(scope: str) -> pd.DataFrame:
    """Recomputes A8 (kpi_health_score) folding in the patient-billed COGS.

    "nonbillable" (today's default, never reaches this function) leaves
    consumption_cost/turnover_annualized exactly as ETL wrote them, fact_consumption
    -only. "billable" replaces consumption_cost with billed cost from
    sales_by_material ('cost' = TOTALCOSTPRICE); "both" SUMS the internal and billed
    COST before recomputing turnover_annualized -- cost+cost, the mirror of
    doh_scoped's qty+qty rule, never mixing the two units.

    Annualization uses ANN=2.0 (6mo -> annual), the same convention
    /kpi/inventory-turnover-ratio/insights already hardcodes for its own live
    aggregates -- not transforms.py's slightly different dynamic `months` factor,
    which is within a rounding error of 2.0 over this dataset's real ~6-month window.
    This keeps the recompute symmetric with the endpoint that reads it.

    health_score/health_tier are NOT recomputed -- that composite (aging + turnover +
    movement) is a separate KPI this Scope toggle doesn't claim to redefine.

    Caveat surfaced in the UI, not hidden here: sales_by_material carries no plant
    dimension, so billed COGS is ONE network-wide figure per material applied to
    every plant that stocks it -- see doh_scoped's identical caveat.
    """
    hs = load("kpi_health_score").copy()
    billed_cogs = pd.Series(dtype=float)
    if scope in ("billable", "both"):
        sm_path = KPI / "sales_by_material.parquet"
        if sm_path.exists():
            sm = pd.read_parquet(sm_path, columns=["material", "cost"])
            billed_cogs = sm.groupby("material")["cost"].sum()
    hs["material"] = hs["material"].astype(str)
    hs["billed_cogs"] = hs["material"].map(billed_cogs).fillna(0.0)
    if scope == "billable":
        hs["consumption_cost"] = hs["billed_cogs"]
    elif scope == "both":
        hs["consumption_cost"] = hs["consumption_cost"] + hs["billed_cogs"]
    ANN = 2.0
    hs["turnover_annualized"] = np.where(hs["closing_stock_value"] > 0,
                                         hs["consumption_cost"] * ANN / hs["closing_stock_value"], 0.0)
    return hs.drop(columns=["billed_cogs"])


def health_scoped(scope: str) -> pd.DataFrame:
    """Recomputes A8's health_score/health_tier composite on top of itr_scoped's
    billed-aware consumption_cost/turnover_annualized.

    itr_scoped's own docstring explicitly deferred this: "health_score/health_tier
    are NOT recomputed -- that composite is a separate KPI this Scope toggle doesn't
    claim to redefine." This function is that follow-up, reproducing transforms.py's
    exact aging_score/turn_score/move_score weights (0.4/0.4/0.2) and health_tier cut
    points unchanged -- only the turnover/movement halves shift with Scope, aging_score
    never does (aging_days is a physical-stock fact).

    Only ever called for scope in ("billable", "both") -- see itr_scoped's own
    "nonbillable" short-circuit note; the "nonbillable" default path never reaches
    this function at the route level, so its zero-drift guarantee is unaffected.
    """
    hs = itr_scoped(scope)  # consumption_cost/turnover_annualized already billed-aware
    aging_score = (1 - (hs["aging_days"].clip(0, 365) / 365)) * 100
    turn_score = (hs["turnover_annualized"].clip(0, 6) / 6) * 100
    move_score = np.where(hs["consumption_cost"] > 0, 100, 0)
    hs["health_score"] = (0.4 * aging_score + 0.4 * turn_score + 0.2 * move_score).round(1)
    hs["health_tier"] = pd.cut(hs["health_score"], [-1, 40, 70, 101],
                               labels=["At Risk", "Watch", "Healthy"]).astype(str)
    return hs


def risk_scoped(scope: str) -> pd.DataFrame:
    """Recomputes A10 (kpi_risk_classification) folding in the patient-billed
    consumption signal into the `consumed` flag.

    A9 (Non-Moving) and A10 (Risk Classification) are built in transforms.py from the
    IDENTICAL `consumed_materials` set -- the exact same bug, two different outputs.
    nonmoving_scoped already fixes A9's side; this is A10's. Keeps aging_days,
    days_to_expiry, nearest_expiry and closing_stock_value exactly as precomputed --
    those are physical-stock/expiry facts Scope has no claim over -- and only
    recomputes `consumed` and the risk_level classifier that reads it (verbatim from
    transforms.py's `_risk()`).
    """
    rc = load("kpi_risk_classification").copy()
    cons = load("fact_consumption")
    moving = set()
    if scope in ("billable", "both"):
        sm_path = KPI / "sales_by_material.parquet"
        if sm_path.exists():
            moving |= set(pd.read_parquet(sm_path)["material"].astype(str).unique())
    if scope == "both":
        moving |= set(cons["material"].astype(str).unique())
    rc["material"] = rc["material"].astype(str)
    rc["consumed"] = rc["material"].isin(moving)

    def _risk(r):
        if pd.notna(r["days_to_expiry"]) and r["days_to_expiry"] <= 90:
            return "High"
        if r["aging_days"] > 365 or not r["consumed"]:
            return "High"
        if r["aging_days"] > 180:
            return "Medium"
        return "Low"
    rc["risk_level"] = rc.apply(_risk, axis=1)
    return rc


@lru_cache(maxsize=1)
def _name_to_code() -> dict:
    dp = load("dim_plant")
    return {str(n): str(c) for c, n in zip(dp["plant"], dp["plant_name"])}


def resolve_plant(region: Optional[str]) -> Optional[str]:
    """Accept a plant code or a hospital name; return the plant code (None => all)."""
    if not region or str(region).upper() in ("ALL", "ALL PLANTS", ""):
        return None
    codes = set(load("dim_plant")["plant"].astype(str))
    if str(region) in codes:
        return str(region)
    return _name_to_code().get(str(region))


def _month_sort_key(df: pd.DataFrame) -> pd.Series:
    if "month" in df.columns:
        return df["month"].map({m: i for i, m in enumerate(MONTH_ORDER)}).fillna(13)
    return pd.Series(np.zeros(len(df)), index=df.index)


def resolve_category(category: Optional[str]) -> Optional[str]:
    """Accept a bucket name or a raw SAP material-type code; return the canonical
    bucket (None => all categories, i.e. no filtering).

    Sibling of resolve_plant, same conventions: empty / "All" => None. An
    unrecognised value is returned unchanged rather than silently ignored, so
    filter_category yields an EMPTY result instead of quietly handing back the
    whole portfolio under a filter label the user believes is applied. Drive the
    selector off GET /meta/categories and this branch never fires.
    """
    if not category:
        return None
    s = str(category).strip()
    if not s or s.upper() in ("ALL", "ALL CATEGORIES", "ALL ITEMS"):
        return None
    return _CATEGORY_ALIASES.get(s.lower(), s)


def filter_plant(df: pd.DataFrame, plant: Optional[str]) -> pd.DataFrame:
    code = resolve_plant(plant)
    if code and "plant" in df.columns:
        return df[df["plant"].astype(str) == code]
    return df


def _is_derived_category_col(df: pd.DataFrame) -> bool:
    """True only when `category` is OUR material-category column, not a same-named
    column that already existed in the source data.

    kpi_purchase_value ships its own `category` — the PO taxonomy (ANTINEOPLASTIC,
    CYTOTOXIC CHEMOTHERAPY, ~1,360 values). _attach_category correctly refuses to
    overwrite it, but filtering on it anyway matched nothing and made
    /kpi/purchase-value return Rs 0 under any category — while monthly-purchase-value
    returned Rs 236.7 Cr for the same filter, so the two procurement money metrics
    flatly contradicted each other. Checking the vocabulary is self-describing and
    cannot go stale the way a hardcoded table list would.
    """
    vals = pd.Series(df["category"]).dropna().unique()
    if len(vals) == 0:
        return False
    return set(map(str, vals)) <= set(CATEGORIES)


def filter_category(df: pd.DataFrame, category: Optional[str]) -> pd.DataFrame:
    """Sibling of filter_plant for the derived material-category dimension.

    A no-op when no category is passed (the default) — every existing call site
    and every number already on the dashboard is unchanged — a no-op on frames that
    carry no `category` column at all (tables with no material grain), and a no-op
    on frames whose `category` means something else entirely (see above), so a
    material-category cut can never silently zero out an unrelated metric.
    """
    cat = resolve_category(category)
    if cat and "category" in df.columns and _is_derived_category_col(df):
        return df[df["category"].astype(str) == cat]
    return df


def query(table: str, plant: Optional[str] = None, material: Optional[str] = None,
          material_group: Optional[str] = None, material_col: str = "material",
          group_col: str = "material_group", sort_chrono: bool = True,
          category: Optional[str] = None) -> list[dict]:
    """Chart-data query: filter by plant + category + (material | material_group), chrono sort."""
    df = load(table)
    df = filter_plant(df, plant)
    df = filter_category(df, category)
    if material and material != "All Items" and material_col in df.columns:
        mats = [m.strip() for m in str(material).split(",")]
        df = df[df[material_col].astype(str).isin(mats)]
    elif material_group and group_col in df.columns:
        df = df[df[group_col].astype(str) == str(material_group)]
    if sort_chrono and {"year", "month"}.issubset(df.columns):
        df = df.assign(_mk=_month_sort_key(df)).sort_values(["year", "_mk"]).drop(columns="_mk")
    return _clean_records(df)


def chart_series(table: str, plant=None, material=None, material_group=None,
                 group_by: Optional[str] = None, measures: Optional[str] = None,
                 top: Optional[int] = None, row_cap: int = 5000,
                 category: Optional[str] = None,
                 frame: Optional[pd.DataFrame] = None) -> list[dict]:
    """Chart data with optional server-side group-by aggregation (bounded payload).

    `frame` substitutes an ALREADY-LOADED equivalent of `table` — used by the one
    caller that has to serve a pre-aggregated KPI from the richer-grain frame the ETL
    built it from (see kpi_generic._CATEGORY_REGRAIN). Defaults to None, i.e. load the
    table, so every other call site is untouched.
    """
    df = load(table) if frame is None else frame
    df = filter_plant(df, plant)
    df = filter_category(df, category)
    if material and material != "All Items" and "material" in df.columns:
        mats = [m.strip() for m in str(material).split(",")]
        df = df[df["material"].astype(str).isin(mats)]
    elif material_group and "material_group" in df.columns:
        df = df[df["material_group"].astype(str) == str(material_group)]

    if group_by:
        gb = [c for c in group_by.split(",") if c in df.columns]
        if measures:
            meas = [c for c in measures.split(",") if c in df.columns]
        else:
            meas = [c for c in df.columns if df[c].dtype.kind in "fiu" and c not in gb]
        if gb and meas:
            df = df.groupby(gb, as_index=False, observed=True)[meas].sum()
            if {"year", "month"}.issubset(df.columns):
                df = df.assign(_mk=_month_sort_key(df)).sort_values(["year", "_mk"]).drop(columns="_mk")
            elif top:
                df = df.nlargest(int(top), meas[0])
            elif meas:
                df = df.sort_values(meas[0], ascending=False)
    elif top and df.select_dtypes("number").shape[1]:
        m0 = df.select_dtypes("number").columns[0]
        df = df.nlargest(int(top), m0)
    elif {"year", "month"}.issubset(df.columns):
        df = df.assign(_mk=_month_sort_key(df)).sort_values(["year", "_mk"]).drop(columns="_mk")

    return _clean_records(df.head(row_cap))


def summarize(table: str, plant=None, material=None, material_group=None,
              category: Optional[str] = None,
              frame: Optional[pd.DataFrame] = None) -> dict:
    """Correct, uncapped sum/mean/count + distinct counts over the filtered table.

    `frame` substitutes an already-loaded equivalent of `table` — see chart_series.
    """
    df = load(table) if frame is None else frame
    df = filter_plant(df, plant)
    df = filter_category(df, category)
    if material and material != "All Items" and "material" in df.columns:
        df = df[df["material"].astype(str).isin([m.strip() for m in str(material).split(",")])]
    elif material_group and "material_group" in df.columns:
        df = df[df["material_group"].astype(str) == str(material_group)]
    out: dict = {"row_count": int(len(df))}
    for c in df.columns:
        if df[c].dtype.kind in "fiu":
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = {"sum": _jsonable(s.sum()),
                      "mean": _jsonable(s.mean()) if len(s) else 0.0,
                      "median": _jsonable(s.median()) if len(s) else 0.0}
        else:
            out[c] = {"distinct": int(df[c].nunique())}
    return out


def _jsonable(v) -> Optional[float]:
    """NaN -> None. A statistic over an all-null column is UNDEFINED, not zero.

    Starlette serialises with allow_nan=False, so a NaN here has always been a hard 500
    rather than a bad number — which is why this can never change an existing response:
    any path that reaches it with a NaN is currently failing outright. It starts
    mattering once a category cut can empty a column that is populated portfolio-wide:
    fact_grn records no PR→GR turnaround at all on onco receipts, so
    /kpi/procurement-cycle-time/summary?Category=Onco Drugs has a fully-null
    avg_pr_to_gr_tat. `null` says "no PR→GR data in this bucket"; 0.0 would say
    "these arrive the same day they are requisitioned", which is a different and false
    claim.
    """
    f = float(v)
    return None if f != f else f


def _apply_filter_protocol(df: pd.DataFrame, params: dict, col_map: dict) -> pd.DataFrame:
    """Apply filter_field_i/operator_i/value_i + global_filter (LIKE / numeric range)."""
    i = 0
    while f"filter_field_{i}" in params:
        field = params.get(f"filter_field_{i}")
        value = params.get(f"filter_value_{i}")
        i += 1
        col = col_map.get(field, field)
        if not value or col not in df.columns:
            continue
        v = str(value).strip()
        if "," in v:  # numeric range "a,b" / "a," / ",b"
            a, b = (p.strip() for p in v.split(",", 1))
            num = pd.to_numeric(df[col], errors="coerce")
            if a and b:
                df = df[(num >= float(a)) & (num <= float(b))]
            elif a:
                df = df[num >= float(a)]
            elif b:
                df = df[num <= float(b)]
        else:
            df = df[df[col].astype(str).str.contains(v, case=False, na=False)]
    gf = params.get("global_filter")
    if gf:
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask |= df[c].astype(str).str.contains(str(gf), case=False, na=False)
        df = df[mask]
    return df


def paginate(table: str, plant: Optional[str], params: dict, col_map: dict,
             columns: Optional[list[str]] = None, rename: Optional[dict] = None,
             category: Optional[str] = None,
             frame: Optional[pd.DataFrame] = None) -> dict:
    """Server-side table: filter + sort + page. Returns {data, total}.

    `frame` substitutes an already-loaded equivalent of `table` — see chart_series.
    """
    df = load(table) if frame is None else frame
    df = filter_plant(df, plant)
    df = filter_category(df, category)
    df = _apply_filter_protocol(df, params, col_map)

    total = len(df)

    sort_field = params.get("sort_field")
    sort_order = (params.get("sort_order") or "asc").lower()
    sort_col = col_map.get(sort_field, sort_field) if sort_field else None
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=(sort_order != "desc"),
                            kind="mergesort", na_position="last")
    elif {"year", "month"}.issubset(df.columns):
        df = df.assign(_mk=_month_sort_key(df)).sort_values(["year", "_mk"]).drop(columns="_mk")

    page = int(params.get("page", 0) or 0)
    page_size = int(params.get("page_size", 25) or 25)
    page_df = df.iloc[page * page_size: page * page_size + page_size]

    if columns:
        page_df = page_df[[c for c in columns if c in page_df.columns]]
    if rename:
        page_df = page_df.rename(columns=rename)
    return {"data": _clean_records(page_df), "total": int(total)}


def _clean_records(df: pd.DataFrame) -> list[dict]:
    df = df.replace({np.nan: None})
    recs = df.to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and (v != v):
                r[k] = None
    return recs
