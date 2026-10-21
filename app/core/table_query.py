"""Server-side tables: facet filters, literal search, sort, page, totals and the CSV export's
projection — the one place every list endpoint behind a KpiTable ends (via
data_access.paginate), so ~45 tables share a single contract:

    f.<field>=<value>              exact match on a facet's stored value; repeat the key for OR
                                   within one field; different fields AND together
    f.<field>.from / .to           inclusive range on a range facet (ISO date or number)
    fq.<field>=<text>              narrows that facet's option list (typeahead)
    facets=1 | only                adds {facets, scope} to the answer; `only` sends no rows
    global_filter=<text>           literal, case-insensitive search ...
    search_fields=a,b              ... in these columns only (default: every column)
    fields=a,b                     rows carry only these columns (the CSV export)
    filter_field_i/filter_value_i  the old per-column protocol, literal too
    page, page_size, sort_field, sort_order

Facet counts are the rows matching the scope, the search and every OTHER facet's selection —
not the facet's own — so ticking a second value of one field never empties its list (standard
faceted search). Without facets=1 the answer is exactly the old {data,total[,sums]}.

Field names in search_fields / fields / sort_field are the names the rows carry (after a
caller's rename); a facet is addressed by its own `field`, the column it filters.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.core import data_access as da

PERIOD = "period"
BLANK_LABEL = "(blank)"
ALL_UNITS, NETWORK, ALL_MONTHS = "all-units", "network", "all-months"
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_NUM = {m: i + 1 for i, m in enumerate(da.MONTH_ORDER)}
_PLAIN_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class Facet:
    """One filter a table offers. `field` is the column it filters and the name it is
    addressed by (f.<field>); `value_label` turns a stored value into words."""
    field: str
    label: str
    kind: Literal["multi", "range"] = "multi"
    value_label: Optional[Callable[[str], str]] = None
    order: Literal["count", "label", "newest"] = "count"
    limit: int = 100


# Filters many tables share beyond the standard ones.
MATERIAL_GROUP = Facet("material_group", "Material group")
VENDOR = Facet("vendor_name", "Vendor")


@dataclass(frozen=True)
class ScopeChip:
    """A filter applied upstream of the rows (the page's own filters), echoed so the table
    can say what it holds and the CSV can carry it in its file name (`token`)."""
    param: str
    label: str
    value: str
    value_label: str
    token: str


# ── Labels and chips ──────────────────────────────────────────────────────────
def month_label(period: str) -> str:
    """'2026-08' -> 'Aug-26', HCG's own month heading; anything else is returned as it is."""
    if not _PERIOD_RE.match(str(period)):
        return str(period)
    y, m = period.split("-")
    return f"{MONTH_ABBR[int(m) - 1]}-{y[2:]}"


def yes_no(value: str) -> str:
    return {"True": "Yes", "False": "No"}.get(value, value)


def token(text) -> str:
    """A file-name-safe form of a value: 'CASH(TPA)' -> 'CASH-TPA'."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-")


def _plant_names() -> dict:
    """SAP plant -> the name the app shows (dim_plant), else the HIS name in the hospital map."""
    hm = da.hospital_map()
    names = {str(p): str(n) for p, n in zip(hm["sap_plant"], hm["his_location_name"]) if str(p).strip()}
    try:
        dp = da.load("dim_plant")
    except FileNotFoundError:
        return names
    return {**names, **{str(c): str(n) for c, n in zip(dp["plant"], dp["plant_name"])}}


def plant_label(code: str) -> str:
    return _plant_names().get(code, code)


def his_label(code: str) -> str:
    """A HIS hospital code (KABHK) in words: its SAP plant's name, else its HIS location name."""
    hm = da.hospital_map()
    row = hm[hm["his_code"].astype(str) == code]
    if row.empty:
        return code
    return _plant_names().get(str(row["sap_plant"].iloc[0])) or str(row["his_location_name"].iloc[0]) or code


def chip(param: str, label: str, value, value_label=None, tok: Optional[str] = None) -> ScopeChip:
    return ScopeChip(param, label, str(value), str(value if value_label is None else value_label),
                     tok or token(value))


def plant_chip(code: Optional[str], *, split: bool = True) -> ScopeChip:
    """Plant= as applied. A table with no hospital column says so rather than letting the
    reader believe a hospital filter was applied to network-wide rows."""
    if code is None:
        return ScopeChip("Plant", "Hospital", "ALL", "All hospitals", ALL_UNITS)
    if not split:
        return ScopeChip("Plant", "Hospital", code, "not split by hospital", NETWORK)
    return ScopeChip("Plant", "Hospital", code, plant_label(code), token(code))


def months_chip(months: Sequence[str], every: Sequence[str]) -> ScopeChip:
    """Month= as applied: one month, a list, or every published month."""
    months = list(months)
    if len(months) > 1 and months == list(every):
        return ScopeChip("Month", "Month", "ALL", "All months", ALL_MONTHS)
    return ScopeChip("Month", "Month", ",".join(months), " + ".join(month_label(m) for m in months),
                     "+".join(months))


def category_chip(category: Optional[str], *, applied: bool = True) -> Optional[ScopeChip]:
    """Category= (the derived material category) as applied, or None when none was asked for."""
    if not category:
        return None
    if not applied:
        return ScopeChip("Category", "Category", category, "not split by material category", "all-categories")
    return ScopeChip("Category", "Category", category, category, token(category).lower())


# ── Parameters ────────────────────────────────────────────────────────────────
def _get(params: Mapping, key: str) -> Optional[str]:
    v = params.get(key)
    if isinstance(v, (list, tuple)):
        v = v[-1] if v else None
    return None if v is None else str(v)


def _getlist(params: Mapping, key: str) -> list:
    """Every value of a repeated key. Request query params are a MultiDict (getlist); a plain
    dict (the AI's direct calls, tests) may hold one value or a list."""
    if hasattr(params, "getlist"):
        return [str(v) for v in params.getlist(key)]
    v = params.get(key)
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


def _csv(params: Mapping, key: str) -> Optional[list]:
    v = _get(params, key)
    return None if v is None else [x.strip() for x in v.split(",") if x.strip()]


def _int(params: Mapping, key: str, default: int) -> int:
    v = _get(params, key)
    try:
        return int(v or default)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{key} must be a whole number, not '{v}'")


def _mode(params: Mapping) -> Optional[str]:
    v = (_get(params, "facets") or "").strip().lower()
    if v in ("", "0", "false"):
        return None
    if v in ("1", "true", "only"):
        return v if v == "only" else "rows"
    raise HTTPException(status_code=400, detail=f"facets takes 1 or only, not '{v}'")


def _offered(name: str, facets: Sequence[Facet]) -> Facet:
    for f in facets:
        if f.field == name:
            return f
    allowed = ", ".join(f.field for f in facets) or "nothing"
    raise HTTPException(status_code=400, detail=f"This table cannot be filtered on '{name}'. It filters on: {allowed}")


def selection(params: Mapping, facets: Sequence[Facet]) -> dict:
    """{field: [values]} for multi facets and {field: {from, to}} for range facets, from the
    f.* parameters. A field the table does not offer is a 400 — never silently ignored."""
    sel: dict = {}
    for key in list(params.keys()):
        if not key.startswith("f."):
            continue
        name, bound = key[2:], None
        for b in ("from", "to"):
            if name.endswith(f".{b}"):
                name, bound = name[: -len(b) - 1], b
        f = _offered(name, facets)
        if f.kind == "range" and bound is None:
            raise HTTPException(status_code=400, detail=f"{f.label} is a range: use f.{f.field}.from / f.{f.field}.to")
        if f.kind == "multi" and bound is not None:
            raise HTTPException(status_code=400, detail=f"{f.label} takes values: f.{f.field}=<value>")
        if bound:
            v = (_get(params, key) or "").strip()
            if v:
                sel.setdefault(name, {})[bound] = v
        else:
            sel[name] = list(dict.fromkeys(sel.get(name, []) + _getlist(params, key)))
    return sel


def _narrowing(params: Mapping, facets: Sequence[Facet]) -> dict:
    """{field: text} from fq.<field>= (typeahead inside one facet's option list)."""
    out = {}
    for key in list(params.keys()):
        if key.startswith("fq."):
            f = _offered(key[3:], facets)
            text = (_get(params, key) or "").strip()
            if text:
                out[f.field] = text
    return out


# ── Columns ───────────────────────────────────────────────────────────────────
def with_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add `period` (YYYY-MM) to a table cut by year + month name, so every such table gets
    the same Month filter. Left alone when there is nothing to build it from — except a frame
    with no rows (the page's hospital or category left none), which still gets the column: a
    Month the user already ticked must give an empty table, not a 400 for a vanished filter."""
    if PERIOD in df.columns or not {"year", "month"} <= set(df.columns):
        return df
    ym = pd.to_numeric(df["year"], errors="coerce") * 100 + df["month"].map(_MONTH_NUM)
    keys = ym.dropna().unique()
    if not len(keys) and not df.empty:
        return df
    lut = {k: f"{int(k) // 100:04d}-{int(k) % 100:02d}" for k in keys}
    return df.assign(**{PERIOD: ym.map(lut).fillna("").astype(str)})


def standard_facets(df: pd.DataFrame) -> list:
    """The filters every table gets when it has the column: Month, Hospital, and the derived
    material Category (not a same-named column meaning something else — da._is_derived_category_col).
    A frame with no rows has no vocabulary to tell the two apart; it keeps the Category filter,
    so a Category the user ticked gives an empty table, not a 400 (it matches nothing either way)."""
    out = []
    if PERIOD in df.columns and all(_PERIOD_RE.match(str(v)) for v in df[PERIOD].dropna().unique() if v != ""):
        out.append(Facet(PERIOD, "Month", value_label=month_label, order="newest"))
    if "plant" in df.columns:
        names = _plant_names()
        out.append(Facet("plant", "Hospital", value_label=lambda c: names.get(c, c), order="label"))
    if "category" in df.columns and (df.empty or da._is_derived_category_col(df)):
        out.append(Facet("category", "Category"))
    return out


def _text(s: pd.Series) -> pd.Series:
    """A column as the strings facets match and count: what the cell prints as, '' for a blank."""
    if pd.api.types.is_string_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        return s.fillna("").astype(str)
    return s.astype(object).where(s.notna(), "").astype(str)


def _bound(s: pd.Series, raw: str, f: Facet, upper: bool):
    """A range end as a comparable value. A plain date as the upper end keeps its whole day."""
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            ts = pd.Timestamp(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{f.label}: '{raw}' is not a date (YYYY-MM-DD)")
        if s.dt.tz is not None and ts.tz is None:
            ts = ts.tz_localize(s.dt.tz)
        return ts + pd.Timedelta(days=1) if upper and _PLAIN_DATE.match(raw) else ts
    try:
        return float(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{f.label}: '{raw}' is not a number")


def _range_mask(s: pd.Series, cond: dict, f: Facet) -> np.ndarray:
    """Rows inside an inclusive range; a blank cell is never inside one."""
    m = np.ones(len(s), dtype=bool)
    v = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_numeric(s, errors="coerce")
    if cond.get("from"):
        m &= (v >= _bound(s, cond["from"], f, False)).fillna(False).to_numpy(dtype=bool)
    if cond.get("to"):
        hi = _bound(s, cond["to"], f, True)
        within = (v < hi) if isinstance(hi, pd.Timestamp) and _PLAIN_DATE.match(cond["to"]) else (v <= hi)
        m &= within.fillna(False).to_numpy(dtype=bool)
    return m


# ── Filtering ─────────────────────────────────────────────────────────────────
def legacy_filters(df: pd.DataFrame, params: Mapping, names: Mapping) -> pd.DataFrame:
    """filter_field_i / filter_value_i: a literal contains-match, or a numeric range 'a,b' /
    'a,' / ',b'. Kept for callers older than the f.* facets."""
    i = 0
    while f"filter_field_{i}" in params:
        field, value = _get(params, f"filter_field_{i}"), _get(params, f"filter_value_{i}")
        i += 1
        col = names.get(field, field)
        if not value or col not in df.columns:
            continue
        v = value.strip()
        if "," in v:
            try:
                lo, hi = (float(p) if p.strip() else None for p in v.split(",", 1))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"filter_value_{i - 1}: '{v}' is not a number range")
            num = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            keep = np.ones(len(df), dtype=bool)
            if lo is not None:
                keep &= num >= lo
            if hi is not None:
                keep &= num <= hi
            df = df[keep]
        else:
            df = df[df[col].astype(str).str.contains(v, case=False, na=False, regex=False).to_numpy()]
    return df


def search(df: pd.DataFrame, text: Optional[str], cols: Sequence[str]) -> Optional[np.ndarray]:
    """Rows whose cells in `cols` contain `text`, literally and ignoring case (None = no
    search). Literal, so '(TPA' or 'CASH(TPA)' find what they say instead of failing as a
    regular expression."""
    if not text:
        return None
    m = np.zeros(len(df), dtype=bool)
    for c in cols:
        m |= df[c].astype(str).str.contains(text, case=False, na=False, regex=False).to_numpy(dtype=bool)
    return m


def apply(df: pd.DataFrame, sel: dict, facets: Sequence[Facet], texts: dict) -> dict:
    """{field: row mask} for each facet with a selection. `texts` holds each multi facet's
    column as _text strings, built once per request. Masks are positional (numpy), so a frame
    with a repeated index label is filtered row for row."""
    by = {f.field: f for f in facets}
    out = {}
    for field, cond in sel.items():
        f = by[field]
        if f.kind == "range":
            if cond:
                out[field] = _range_mask(df[field], cond, f)
        elif cond:
            out[field] = texts[field].isin(cond).to_numpy(dtype=bool)
    return out


def _all(n: int, masks) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    for x in masks:
        if x is not None:
            m &= x
    return m


# ── Facet options ─────────────────────────────────────────────────────────────
def _multi(f: Facet, vals: pd.Series, selected: list, narrow: Optional[str]) -> dict:
    counts = vals.value_counts()
    label = f.value_label or (lambda v: v)
    words = lambda v: BLANK_LABEL if v == "" else label(v)  # noqa: E731
    items = {v: int(c) for v, c in counts.items() if c > 0}
    for v in selected:
        items.setdefault(v, 0)
    if narrow:
        n = narrow.lower()
        items = {v: c for v, c in items.items() if v in selected or n in v.lower() or n in words(v).lower()}
    if f.order == "newest":
        ranked = sorted((v for v in items if v != ""), reverse=True) + [v for v in items if v == ""]
    elif f.order == "label":
        ranked = sorted(items, key=lambda v: (v == "", _natural(words(v))))
    else:
        ranked = sorted(items, key=lambda v: (-items[v], v == "", words(v).lower()))
    listed = ranked[: f.limit]
    listed += [v for v in selected if v not in listed]
    shown = set(listed)
    return {"field": f.field, "label": f.label, "kind": "multi",
            "values": [{"value": v, "label": words(v), "count": items[v]} for v in listed],
            "more": sum(v not in shown for v in ranked), "selected": list(selected)}


def _natural(text: str) -> list:
    """Sort key that reads numbers as numbers: '31-90' before '181-365', 'Unit 9' before 'Unit 10'."""
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", text.lower()) if t]


def _edge(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return int(v) if float(v).is_integer() else float(v)


def _range(f: Facet, s: pd.Series, selected: Optional[dict]) -> dict:
    v = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_numeric(s, errors="coerce")
    v = v.dropna()
    return {"field": f.field, "label": f.label, "kind": "range",
            "range": {"min": _edge(v.min()) if len(v) else None, "max": _edge(v.max()) if len(v) else None},
            "selected": selected or {}}


def options(df: pd.DataFrame, facets: Sequence[Facet], sel: dict, masks: dict, texts: dict,
            searched: Optional[np.ndarray], narrow: dict) -> list:
    """Each facet's values and counts over the rows matching the search and every OTHER
    facet's selection; selected values are always listed (count 0 when nothing matches)."""
    out = []
    for f in facets:
        m = _all(len(df), [searched, *(x for k, x in masks.items() if k != f.field)])
        if f.kind == "range":
            out.append(_range(f, df[f.field][m], sel.get(f.field)))
        else:
            out.append(_multi(f, texts[f.field][m], sel.get(f.field, []), narrow.get(f.field)))
    return out


# ── The page ──────────────────────────────────────────────────────────────────
def column_sums(df: pd.DataFrame, sum_cols: Sequence[str], columns: Optional[Sequence[str]],
                rename: Mapping) -> dict:
    """{column: Σ} over every filtered row (not one page), keyed by the name the rows carry. A
    column the table does not return is skipped; an all-null column is None, not 0 (see
    da._jsonable). Rounded to 4 dp only to drop float noise (0.1 + 0.2) — the rows carry 2 dp
    for money, 3 for quantity. A count column (lines) stays a whole number."""
    out = {}
    for c in sum_cols:
        if c not in df.columns or (columns and c not in columns):
            continue
        key = rename.get(c, c)
        if pd.api.types.is_integer_dtype(df[c]):
            out[key] = int(df[c].sum())
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        v = float(s.sum(min_count=1)) if len(s) else 0.0
        out[key] = None if v != v else round(v, 4) + 0.0
    return out


def _merge(standard: list, declared: Sequence[Facet], df: pd.DataFrame) -> list:
    """Standard facets first (Month, Hospital, Category), then the table's own; a table's own
    facet replaces a standard one of the same field. Only columns the frame has are offered."""
    own = {f.field for f in declared}
    return [f for f in (*(s for s in standard if s.field not in own), *declared) if f.field in df.columns]


def table_page(df: pd.DataFrame, params: Mapping, *, facets: Sequence[Facet] = (),
               scope: Sequence[Optional[ScopeChip]] = (), columns: Optional[Sequence[str]] = None,
               rename: Optional[Mapping] = None, sum_cols: Optional[Sequence[str]] = None,
               col_map: Optional[Mapping] = None) -> dict:
    """Filter, search, sort and page `df` (already cut to the page's scope).

    `columns` / `rename` pick and rename what the rows carry; `col_map` maps a requested field
    name to its column (default: the reverse of `rename`). `sum_cols` adds `sums` over the
    filtered rows. `facets` are the table's own filters beside the standard ones; `scope` the
    upstream filters echoed with facets=1 (None entries are skipped)."""
    rename = dict(rename or {})
    names = {**{v: k for k, v in rename.items()}, **(col_map or {})}
    mode = _mode(params)

    df = legacy_filters(df, params, names)
    own_cols = list(df.columns)
    # The filters are worked out only when asked for (facets=1, or an f.* / fq.* parameter): a
    # plain page — the top-5 tiles, the totals rows, the AI's reads — pays nothing for them.
    asked = mode is not None or any(k.startswith(("f.", "fq.")) for k in params.keys())
    if asked:
        df = with_period(df)
    offered = _merge(standard_facets(df), facets, df) if asked else []
    sel = selection(params, offered)
    narrow = _narrowing(params, offered)

    wanted = _csv(params, "search_fields")
    cols = own_cols if wanted is None else [c for c in (names.get(x, x) for x in wanted) if c in df.columns]
    searched = search(df, _get(params, "global_filter"), cols)
    texts = {f.field: _text(df[f.field]) for f in offered
             if f.kind == "multi" and (f.field in sel or mode is not None)}
    masks = apply(df, sel, offered, texts)
    rows = df[_all(len(df), [searched, *masks.values()])]

    out: dict = {"data": [], "total": int(len(rows))}
    if sum_cols is not None:
        out["sums"] = column_sums(rows, sum_cols, columns, rename)
    if mode is not None:
        out["facets"] = options(df, offered, sel, masks, texts, searched, narrow)
        out["scope"] = [asdict(c) for c in scope if c is not None]
    if mode == "only":
        return out

    sort_field, order = _get(params, "sort_field"), (_get(params, "sort_order") or "asc").lower()
    sort_col = names.get(sort_field, sort_field) if sort_field else None
    if sort_col and sort_col in rows.columns:
        rows = rows.sort_values(sort_col, ascending=(order != "desc"), kind="mergesort", na_position="last")
    elif {"year", "month"} <= set(rows.columns):
        rows = rows.assign(_mk=da._month_sort_key(rows)).sort_values(["year", "_mk"]).drop(columns="_mk")

    page, size = _int(params, "page", 0), _int(params, "page_size", 25) or 25
    rows = rows.iloc[page * size: page * size + size]
    rows = rows[[c for c in (columns or own_cols) if c in rows.columns]].rename(columns=rename)
    fields = _csv(params, "fields")
    if fields is not None:
        rows = rows[[c for c in dict.fromkeys(fields) if c in rows.columns]]
    out["data"] = da._clean_records(rows)
    return out
