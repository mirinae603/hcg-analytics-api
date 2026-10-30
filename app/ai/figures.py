"""The procurement totals the AI's texts quote, read from the data when a request is sent.

The prompts used to state them outright ("₹649.91 Cr", "₹48.79 Cr of GST", "₹1,777.65 Cr of
retail value"), and the import rebuild that moved every one of them left the model quoting
totals the dashboard no longer shows. The texts now carry tokens that windows.fill() puts the
current figures into, beside the months:

  <<PROCUREMENT_TOTAL>>   the dashboard's procurement spend: SUM(kpi_purchase_value.purchase_value)
  <<PROCUREMENT_GST>>     the GST beside it: SUM(fact_po cgst_value + sgst_value + igst_value)
  <<RECEIVED_TOTAL>>      goods received at cost: SUM(fact_grn.total_amount_wo_tax)
  <<RECEIVED_MRP>>        their retail value: SUM(fact_grn.total_mrp_value)
  <<GRN_DISCOUNT>>        SUM(fact_grn.discount_value)
  <<GRN_DISCOUNT_LINES>>  the share of receipt lines that carry a discount ("4.1%")

Money is written in crore as the texts do ("₹806.91 Cr"). A table that cannot be read gives a
neutral phrase, never a number; each value is cached against its file's mtime, so a rebuilt
table changes the text on the next call.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

from app.core.config import settings

CRORE = 1e7


class Total(NamedTuple):
    """One quoted total: the columns of one table added up."""
    folder: str                 # the settings attribute of the table's folder
    table: str
    columns: tuple[str, ...]
    neutral: str                # the words used when the table cannot be read


TOTALS = {
    "PROCUREMENT_TOTAL": Total("KPI_DIR", "kpi_purchase_value", ("purchase_value",),
                               "the procurement total"),
    "PROCUREMENT_GST": Total("CURATED_DIR", "fact_po", ("cgst_value", "sgst_value", "igst_value"),
                             "the GST on those orders"),
    "RECEIVED_TOTAL": Total("CURATED_DIR", "fact_grn", ("total_amount_wo_tax",), "the value received"),
    "RECEIVED_MRP": Total("CURATED_DIR", "fact_grn", ("total_mrp_value",), "the retail value received"),
    "GRN_DISCOUNT": Total("CURATED_DIR", "fact_grn", ("discount_value",), "the discounts received"),
}
DISCOUNT_LINES = Total("CURATED_DIR", "fact_grn", ("discount_value",), "a small share")


def _path(t: Total) -> Path:
    return Path(getattr(settings, t.folder)) / f"{t.table}.parquet"


def _stamp(path: Path) -> Optional[tuple]:
    try:
        st = path.stat()
    except OSError:
        return None
    return str(path), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=16)
def _sum_at(path: str, _mtime: int, _size: int, columns: tuple[str, ...]) -> Optional[float]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=list(columns))
    except Exception:  # unreadable or wrong schema: the figure is unknown
        return None
    return float(sum(pc.sum(t.column(c)).as_py() or 0.0 for c in columns))


@lru_cache(maxsize=4)
def _nonzero_share_at(path: str, _mtime: int, _size: int, column: str) -> Optional[float]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        col = pq.read_table(path, columns=[column]).column(column)
    except Exception:
        return None
    if not len(col):
        return None
    nonzero = pc.sum(pc.not_equal(pc.fill_null(col, 0), 0)).as_py() or 0
    return nonzero / len(col) * 100


def total(key: str) -> Optional[float]:
    """TOTALS[key] in rupees; None when its table cannot be read."""
    t = TOTALS[key]
    k = _stamp(_path(t))
    return _sum_at(*k, t.columns) if k else None


def crore(v: float) -> str:
    """806908357.4 -> '₹80.69 Cr'."""
    return f"₹{v / CRORE:,.2f} Cr"


def values() -> dict[str, str]:
    """Every token's text for the data as it is now."""
    out = {}
    for key, t in TOTALS.items():
        v = total(key)
        out[key] = crore(v) if v is not None else t.neutral
    k = _stamp(_path(DISCOUNT_LINES))
    share = _nonzero_share_at(*k, DISCOUNT_LINES.columns[0]) if k else None
    out["GRN_DISCOUNT_LINES"] = f"{share:.1f}%" if share is not None else DISCOUNT_LINES.neutral
    return out
