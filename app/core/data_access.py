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


def _read_parquet(table: str) -> pd.DataFrame:
    for base in (KPI, CURATED):
        p = base / f"{table}.parquet"
        if p.exists():
            return pd.read_parquet(p)
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


def filter_plant(df: pd.DataFrame, plant: Optional[str]) -> pd.DataFrame:
    code = resolve_plant(plant)
    if code and "plant" in df.columns:
        return df[df["plant"].astype(str) == code]
    return df


def query(table: str, plant: Optional[str] = None, material: Optional[str] = None,
          material_group: Optional[str] = None, material_col: str = "material",
          group_col: str = "material_group", sort_chrono: bool = True) -> list[dict]:
    """Chart-data query: filter by plant + (material | material_group), chrono sort."""
    df = load(table)
    df = filter_plant(df, plant)
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
                 top: Optional[int] = None, row_cap: int = 5000) -> list[dict]:
    """Chart data with optional server-side group-by aggregation (bounded payload)."""
    df = load(table)
    df = filter_plant(df, plant)
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


def summarize(table: str, plant=None, material=None, material_group=None) -> dict:
    """Correct, uncapped sum/mean/count + distinct counts over the filtered table."""
    df = load(table)
    df = filter_plant(df, plant)
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
             columns: Optional[list[str]] = None, rename: Optional[dict] = None) -> dict:
    """Server-side table: filter + sort + page. Returns {data, total}."""
    df = load(table)
    df = filter_plant(df, plant)
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
