"""Data exports — HIS pharmacy sales by item (IP and OP) and SAP goods receipts, as paged tables.

    GET /export/meta    -> both datasets: the months each holds and its rows per month, its
                           columns (field, label, kind), where the rows come from and what
                           they leave out
    GET /export/sales   -> {data,total,sums,facets,scope}: one row per sales_items row — month x
                           hospital x IP/OP x payer x item x formulary class — with units, cost,
                           revenue (MRP) and margin
    GET /export/grn     -> {data,total,sums,facets,scope}: one row per goods-receipt line of the
                           curated fact_grn, with the plant's name from dim_plant

Both lists sit on the shared table contract (formulary._page -> data_access.paginate ->
app.core.table_query), so a KpiTable's Export CSV pages them 5,000 rows at a time with fields=
and writes every row the filters match: no download code of their own. The page filters (Plant,
Month; on sales also Type, Segment and Payer) cut the rows first and come back as `scope`; the
table's own facets and search narrow them after. Rows come in a fixed order (sales: month,
hospital, item; GRN: month, plant, receipt date, item) and sorting is stable, so the pages of
one export never overlap or skip a row.

What they leave out: neither table holds patient, bill, invoice, GR or PO numbers, nor doctor
names. The formulary reader drops the patient columns before anything is summed and the GRN
reader never reads a document number (app/etl/formulary/config.py, app/etl/ingest.py). The
frames below also pick their columns by name, so a column added to a source later is not
exported until it is listed here.

Memory: fact_grn holds ~3.8 lakh lines. It is read once per file version — only the columns
exported, text kept as categories (~40 MB instead of ~110 MB) — and cut per request; Month=ALL
with every hospital pages the cached frame itself, so no second copy is made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Request

from app.api import formulary as F
from app.api import revenue as R
from app.api import spend as S
from app.core import table_query as tq
from app.core.config import settings
from app.etl.formulary import config as FC

router = APIRouter()

# Sales by item (FC.SALES_ITEMS, contract C1) come from the formulary build, in
# FORMULARY_DATA_DIR; goods receipts and plant names from the curated tables run_etl writes.
GRN, PLANTS = "fact_grn", "dim_plant"
REBUILD_SALES = "python -m app.etl.formulary --from-cache"
REBUILD_GRN = "python -m app.etl.run_etl --stage curated"
NOT_BUILT_SALES = f"The sales-by-item table has not been built yet — run {REBUILD_SALES}"
NOT_BUILT_GRN = f"The GRN table has not been built yet — run {REBUILD_GRN}"


class Column(NamedTuple):
    field: str
    label: str
    kind: str                      # the UI's column kinds: inr | num | text | date


# What each export's rows carry, in order. Money is rounded to the paisa and quantities to 3
# dp, as the neighbouring tables do.
SALES_COLUMNS = (
    Column("period", "Month", "text"),
    Column("plant", "SAP plant", "text"),
    Column("hospital_name", "Hospital", "text"),
    Column("patient_type", "IP / OP", "text"),
    Column("payer_bucket", "Payer", "text"),
    Column("material", "Material code", "text"),
    Column("material_desc", "Item", "text"),
    Column("generic_name", "Generic name", "text"),
    Column("manufacturer", "Manufacturer", "text"),
    Column("category", "Category", "text"),
    Column("formulary_class", "Formulary class", "text"),
    Column("bucket_label", "Formulary bucket", "text"),
    Column("qty", "Units", "num"),
    Column("cost", "Cost", "inr"),
    Column("mrp", "Revenue (MRP)", "inr"),
    Column("margin", "Margin", "inr"),
    Column("lines", "Lines", "num"),
    # Lines whose cost price was blank (counted as cost 0, as HCG's sheet does) and their MRP,
    # beside the margin they inflate. Only when the build could tell (contract C1).
    Column("cost_missing_lines", "Lines with no cost price", "num"),
    Column("mrp_cost_missing", "MRP of lines with no cost price", "inr"),
)
SALES_OPTIONAL = ("cost_missing_lines", "mrp_cost_missing")
SALES_COUNTS = ("lines", "cost_missing_lines")
# Read from sales_items: the stored columns above, and the group the bucket comes from.
SALES_READ = tuple(c.field for c in SALES_COLUMNS if c.field not in ("bucket_label", "margin")) + ("formulary_group",)
SALES_ORDER = ["period", "plant", "material", "patient_type", "payer_bucket", "formulary_class"]
SALES_SUMS = ["qty", "cost", "mrp", "margin", "lines", "cost_missing_lines", "mrp_cost_missing"]
SALES_FACETS = (tq.Facet("patient_type", "Patient type", order="label"), tq.Facet("category", "Category"),
                tq.Facet("bucket_label", "Formulary bucket"), tq.Facet("payer_bucket", "Payer", order="label"))

GRN_COLUMNS = (
    Column("month", "Month", "text"),
    Column("plant", "SAP plant", "text"),
    Column("plant_name", "Plant", "text"),
    Column("sloc", "Storage location", "text"),
    Column("material", "Material code", "text"),
    Column("material_desc", "Item", "text"),
    Column("major_group", "Major group", "text"),
    Column("minor_group", "Minor group", "text"),
    Column("vendor_code", "Supplier code", "text"),
    Column("vendor_name", "Supplier", "text"),
    Column("gr_qty", "Quantity received", "num"),
    Column("base_price", "Base price", "inr"),
    Column("net_price", "Net price", "inr"),
    Column("unit_mrp", "MRP per unit", "inr"),
    Column("total_amount_wo_tax", "Amount before tax", "inr"),
    Column("tax_pct", "Tax %", "num"),
    Column("tax_amount", "Tax", "inr"),
    Column("total_amount_with_tax", "Amount with tax", "inr"),
    Column("discount_value", "Discount", "inr"),
    Column("total_mrp_value", "MRP value", "inr"),
    Column("po_date", "Order date", "date"),
    Column("gr_date", "Receipt date", "date"),
    Column("expiry_date", "Expiry date", "date"),
)
# fact_grn's year + month_num: the posting date's month, the month every procurement view
# counts a receipt in (legacy_kpi._year_month).
GRN_MONTH_FROM = ("year", "month_num")
GRN_READ = tuple(c.field for c in GRN_COLUMNS if c.field not in ("month", "plant_name")) + GRN_MONTH_FROM
GRN_ORDER = ["month", "plant", "gr_date", "material"]
GRN_SUMS = ["gr_qty", "total_amount_wo_tax", "tax_amount", "total_amount_with_tax", "discount_value",
            "total_mrp_value"]
GRN_FACETS = (tq.Facet("month", "Month", value_label=tq.month_label, order="newest"),
              tq.Facet("major_group", "Major group"), tq.VENDOR)

LABEL_SALES = "IP & OP sales by item"
LABEL_GRN = "Goods receipts (GRN)"
SOURCE_SALES = ("HIS IP + OP pharmacy sales — HCG's monthly consolidated files, classed by the same "
                "month's Item Master (formulary build, sales_items)")
SOURCE_GRN = "SAP goods receipt extracts (curated fact_grn), plant names from dim_plant"
NOTE_SALES = ("One row per month × hospital × IP/OP × payer × item × formulary class, summed from the sales "
              "lines. Revenue is the total MRP, cost the total cost price, margin = revenue − cost. A line "
              "with a blank cost price counts as cost 0, as in HCG's sheet; how many there are and their MRP "
              "are in the last two columns. Consumables are their own bucket; drugs take the bucket the "
              "month's Item Master gives them.")
NOTE_GRN = ("One row per goods-receipt line (SAP movement 101; returns are left out), for every plant in "
            "SAP's extract — not only the units in HCG's purchase scope. Prices are as SAP holds them: a few "
            "lines carry placeholder or mistyped prices, which the dashboard's purchase views clean and this "
            "export does not.")
NOTE_IDS = ("No patient, bill, invoice, GR or PO numbers are kept in these tables, so none are exported; "
            "nor are doctor names.")
NOTE_FILTERS = ("Each export holds exactly the rows its table shows: the page filters (hospital, month and, "
                "on sales, IP/OP, segment and payer) cut the rows first, then the table's own filters and "
                "search.")
NOTE_UNITS = "Units add vials, tablets and pieces together as they were billed or received."


# ── Tables ────────────────────────────────────────────────────────────────────
# Read at CALL time from the settings (never a path captured at import) and keyed on each
# file's mtime/size like the formulary tables: a rebuild — or a test pointing the settings at
# its own fixtures — is picked up without a restart. One version of each is held (maxsize=1).
class Dataset(NamedTuple):
    frame: pd.DataFrame            # the exported columns, in export order, sorted
    months: tuple                  # 'YYYY-MM' the table holds, oldest first


class SalesScope(NamedTuple):
    plant: Optional[str]           # SAP plant code; None => all hospitals
    months: tuple                  # periods 'YYYY-MM'
    ptype: Optional[str]           # 'IP' / 'OP'; None => both
    segment: Optional[str]         # 'onco' / 'non-onco' / 'consumables'; None => all
    payer: Optional[str]           # 'CASH' / 'CASH(TPA)' / 'GOV'; None => all


class GrnScope(NamedTuple):
    plant: Optional[str]
    months: tuple


def _sales_path() -> Path:
    return F._data_dir() / f"{FC.SALES_ITEMS}.parquet"


def _curated(name: str) -> Path:
    return Path(settings.CURATED_DIR) / f"{name}.parquet"


def _version() -> tuple:
    """Identity of the published data — the key of the cached meta."""
    return tuple((str(p), F._stamp(p)) for p in (_sales_path(), _curated(GRN), _curated(PLANTS)))


def _require(path: Path, need, optional=(), *, rebuild: str) -> list:
    """The columns to read: `need` (a 503 naming the gap when one is missing — a table built
    before this export existed) and whichever of `optional` the file holds."""
    have = set(pq.read_schema(path).names)
    missing = [c for c in need if c not in have and c not in optional]
    if missing:
        raise HTTPException(status_code=503, detail=(
            f"{path.stem} has no {', '.join(missing)} column — rebuild it: {rebuild}"))
    return [c for c in need if c in have]


def _categories(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Text columns as categories: a few thousand distinct values instead of one string per row.
    The categories come sorted, so a sort on them is alphabetical."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _round(df: pd.DataFrame, columns) -> pd.DataFrame:
    for c in columns:
        if c.field in df.columns and c.kind in ("inr", "num") and not pd.api.types.is_integer_dtype(df[c.field]):
            df[c.field] = df[c.field].astype(float).round(2 if c.kind == "inr" else 3) + 0.0
    return df


def _counts(month: pd.Series) -> tuple:
    """(months oldest first, {month: rows}) of a categorical month column."""
    n = month.value_counts(sort=False)
    n = n[n > 0].sort_index()
    return tuple(str(m) for m in n.index), {str(m): int(c) for m, c in n.items()}


@lru_cache(maxsize=1)
def _sales_at(path: str, stamp: tuple) -> Dataset:
    df = pd.read_parquet(path, columns=_require(Path(path), SALES_READ, SALES_OPTIONAL, rebuild=REBUILD_SALES))
    for c in SALES_COUNTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    df = _round(df, SALES_COLUMNS)
    # From the rounded amounts, so every row's revenue - cost - margin is exactly 0 in the file.
    df["margin"] = (df["mrp"] - df["cost"]).round(2) + 0.0
    # The revenue page's bucket rule (contract C1), so the export and the page cannot disagree.
    df["bucket_label"] = R.bucket_of(df).map(R.BUCKETS)
    df = _categories(df, [c.field for c in SALES_COLUMNS if c.kind == "text"])
    df = df.sort_values(SALES_ORDER, kind="mergesort").reset_index(drop=True)
    df = df[[c.field for c in SALES_COLUMNS if c.field in df.columns]]
    return Dataset(df, _counts(df["period"])[0])


def _sales() -> Dataset:
    p = _sales_path()
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT_SALES)
    return _sales_at(str(p), stamp)


def _year_month(df: pd.DataFrame) -> pd.Series:
    """'YYYY-MM' from year + month_num (blank when either is)."""
    ym = pd.to_numeric(df[GRN_MONTH_FROM[0]], errors="coerce") * 100 + \
        pd.to_numeric(df[GRN_MONTH_FROM[1]], errors="coerce")
    lut = {k: f"{int(k) // 100:04d}-{int(k) % 100:02d}" for k in ym.dropna().unique()}
    return ym.map(lut)


def _iso_dates(s: pd.Series) -> pd.Series:
    """A date column as 'YYYY-MM-DD' categories (a blank stays blank): the export writes dates
    without a time, and a few hundred distinct days cost next to nothing to hold."""
    c = pd.to_datetime(s, errors="coerce").dt.normalize().astype("category")
    return c.cat.rename_categories(list(c.cat.categories.strftime("%Y-%m-%d")))


def _plant_names(path: str, stamp) -> dict:
    """SAP plant -> its name in dim_plant (empty when dim_plant is missing)."""
    if stamp is None:
        return {}
    dp = pd.read_parquet(path, columns=["plant", "plant_name"])
    return {str(c): str(n) for c, n in zip(dp["plant"], dp["plant_name"])}


@lru_cache(maxsize=1)
def _grn_at(path: str, stamp: tuple, plants: str, plants_stamp) -> Dataset:
    df = pd.read_parquet(path, columns=_require(Path(path), GRN_READ, rebuild=REBUILD_GRN))
    df["month"] = _year_month(df)
    df = df.drop(columns=list(GRN_MONTH_FROM))
    df["plant_name"] = df["plant"].astype(str).map(_plant_names(plants, plants_stamp))
    for c in GRN_COLUMNS:
        if c.kind == "date":
            df[c.field] = _iso_dates(df[c.field])
    df = _round(df, GRN_COLUMNS)
    df = _categories(df, [c.field for c in GRN_COLUMNS if c.kind == "text"])
    df = df.sort_values(GRN_ORDER, kind="mergesort").reset_index(drop=True)
    df = df[[c.field for c in GRN_COLUMNS]]
    return Dataset(df, _counts(df["month"])[0])


def _grn() -> Dataset:
    p, n = _curated(GRN), _curated(PLANTS)
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT_GRN)
    return _grn_at(str(p), stamp, str(n), F._stamp(n))


@lru_cache(maxsize=1)
def _grn_months_at(path: str, stamp: tuple) -> tuple:
    """(months, rows per month) of fact_grn from its two month columns alone, so the export
    page's meta — and a bad Month — never load the whole table."""
    cols = _require(Path(path), GRN_MONTH_FROM, rebuild=REBUILD_GRN)
    return _counts(_year_month(pd.read_parquet(path, columns=cols)).astype("category"))


def _grn_months() -> tuple:
    p = _curated(GRN)
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT_GRN)
    return _grn_months_at(str(p), stamp)


def clear_caches() -> None:
    for c in (_sales_at, _grn_at, _grn_months_at, _meta_cached):
        c.cache_clear()


# ── Arguments ─────────────────────────────────────────────────────────────────
def _sales_scope(Plant, Month, Type, Segment, Category, Payer) -> SalesScope:
    months = _sales().months                # 503 first: nothing below means anything without data
    return SalesScope(plant=F._plant_arg(Plant), months=S._months_arg(Month, list(months), "sales"),
                      ptype=F._choice(Type, {t: t for t in F.TYPES}, "Type"),
                      segment=S._category_arg(Segment, Category),
                      payer=F._choice(Payer, {p: p for p in F.PAYERS}, "Payer"))


def _grn_scope(Plant, Month) -> GrnScope:
    months = _grn_months()[0]               # 503 first
    return GrnScope(plant=F._plant_arg(Plant), months=S._months_arg(Month, list(months), "GRN"))


def _cut(df: pd.DataFrame, every: tuple, sc, month_col: str, **cols) -> pd.DataFrame:
    """Rows of `df` in the page's scope: the plant, the months (unless every one) and `cols`
    ({column: value}; None = no cut). The cached frame itself when nothing cuts it."""
    m = np.ones(len(df), dtype=bool)
    if sc.plant:
        m &= (df["plant"] == sc.plant).to_numpy(dtype=bool)
    if tuple(sc.months) != tuple(every):
        m &= df[month_col].isin(sc.months).to_numpy(dtype=bool)
    for col, value in cols.items():
        if value is not None:
            m &= (df[col] == value).to_numpy(dtype=bool)
    return df if m.all() else df[m]


def _chips(sc, every: tuple, *extra: tq.ScopeChip) -> tuple:
    """The page filters a list was cut to, as table scope chips."""
    return (tq.plant_chip(sc.plant), tq.months_chip(sc.months, every), *extra)


def _built_at(p: Path) -> Optional[str]:
    st = F._stamp(p)
    return datetime.fromtimestamp(st[0] / 1e9, tz=timezone.utc).isoformat(timespec="seconds") if st else None


def _dataset(key: str, label: str, source: str, columns, note: str, params, *, months=(), rows=None,
             built_at=None, options=None) -> dict:
    rows = rows or {}
    return {"key": key, "label": label, "endpoint": f"/export/{key}", "source": source,
            "available": built_at is not None, "built_at": built_at,
            "months": list(months), "month_labels": {m: tq.month_label(m) for m in months},
            "latest": months[-1] if months else None,
            "rows_by_month": rows, "rows": int(sum(rows.values())),
            "columns": [c._asdict() for c in columns], "params": list(params),
            "options": options or {}, "note": note}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/export/meta")
def export_meta():
    return _meta_cached(_version())


@lru_cache(maxsize=4)
def _meta_cached(version: tuple) -> dict:
    # A table that is not built leaves its dataset empty (available: false) with the command
    # that builds it; the other dataset is still listed, so meta never answers 503.
    try:
        sales = _sales()
        s_months, s_rows = _counts(sales.frame["period"])
        s = dict(months=s_months, rows=s_rows, built_at=_built_at(_sales_path()))
        s_cols, s_note = [c for c in SALES_COLUMNS if c.field in sales.frame.columns], NOTE_SALES
    except HTTPException as e:
        s, s_cols, s_note = {}, SALES_COLUMNS, f"{NOTE_SALES} {e.detail}."
    try:
        g_months, g_rows = _grn_months()
        g, g_note = dict(months=g_months, rows=g_rows, built_at=_built_at(_curated(GRN))), NOTE_GRN
    except HTTPException as e:
        g, g_note = {}, f"{NOTE_GRN} {e.detail}."
    built = [b for b in (s.get("built_at"), g.get("built_at")) if b]
    return {
        "datasets": [
            _dataset("sales", LABEL_SALES, SOURCE_SALES, s_cols, s_note,
                     ("Plant", "Month", "Type", "Segment", "Payer"),
                     options={"Type": list(F.TYPES), "Payer": list(F.PAYERS),
                              "Segment": [{"key": k, "label": v} for k, v in S.CATEGORY_LABELS.items()]},
                     **s),
            _dataset("grn", LABEL_GRN, SOURCE_GRN, GRN_COLUMNS, g_note, ("Plant", "Month"), **g),
        ],
        "notes": [NOTE_IDS, NOTE_FILTERS, NOTE_UNITS],
        "data_as_of": max(built) if built else None,
    }


@router.get("/export/sales")
def export_sales(request: Request, Plant: str = Query(None), Month: str = Query(None),
                 Type: str = Query(None), Segment: str = Query(None), Category: str = Query(None),
                 Payer: str = Query(None)):
    sc = _sales_scope(Plant, Month, Type, Segment, Category, Payer)
    data = _sales()
    rows = _cut(data.frame, data.months, sc, "period", patient_type=sc.ptype,
                category=S.CATEGORIES.get(sc.segment), payer_bucket=sc.payer)
    extra = []
    if sc.ptype:
        extra.append(tq.chip("Type", "Patient type", sc.ptype))
    if sc.segment:
        extra.append(tq.chip("Segment", "Segment", sc.segment, S.CATEGORY_LABELS[sc.segment]))
    if sc.payer:
        extra.append(tq.chip("Payer", "Payer", sc.payer))
    return F._page(request, rows, sum_cols=SALES_SUMS, facets=SALES_FACETS,
                   scope=_chips(sc, data.months, *extra))


@router.get("/export/grn")
def export_grn(request: Request, Plant: str = Query(None), Month: str = Query(None)):
    sc = _grn_scope(Plant, Month)
    data = _grn()
    return F._page(request, _cut(data.frame, data.months, sc, "month"), sum_cols=GRN_SUMS,
                   facets=GRN_FACETS, scope=_chips(sc, data.months))
