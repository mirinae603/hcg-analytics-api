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
                 category: Optional[str] = None) -> list[dict]:
    """Chart data with optional server-side group-by aggregation (bounded payload)."""
    df = load(table)
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
              category: Optional[str] = None) -> dict:
    """Correct, uncapped sum/mean/count + distinct counts over the filtered table."""
    df = load(table)
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
            out[c] = {"sum": float(s.sum()), "mean": float(s.mean()) if len(s) else 0.0,
                      "median": float(s.median()) if len(s) else 0.0}
        else:
            out[c] = {"distinct": int(df[c].nunique())}
    return out


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
             category: Optional[str] = None) -> dict:
    """Server-side table: filter + sort + page. Returns {data, total}."""
    df = load(table)
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
