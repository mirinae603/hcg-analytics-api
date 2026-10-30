"""Ingest raw HCG SAP Excel extracts into normalized curated fact tables.

  load_consumption  stores issues (MvT 201): the consolidated extract and the monthly ones,
                    exactly one source per month
  load_inventory    the stock snapshot
  load_grn          goods receipts (MvT 101), through the purchase build's GRN reader
  load_po           purchase-order lines

Each loader returns a SourceRead: a tidy frame with canonical snake_case columns and coerced
dtypes, and a report of what it read and what it left out (counts, values and file names
only), which run_etl writes to curated/_dq_report.json.

Where the extracts are: RAW_DATA_DIR (default: the repo's master/ folder, the root the
formulary and purchase builds use; HCG's files sit one hashed folder below it) is searched for
each source folder by its name. Folder names are not trusted for months: 'GRN Dec 25 to May
2026' also holds Jun-Aug 2026, 'Consumption Dec 25 to May 26' Jun, Jul and Sep 2026.

What the extracts get wrong, and what the loaders do about it:
  - Dates are real dates in some files (and rows) and 'dd.mm.yyyy' text in others. Text is
    read DAY-FIRST, explicitly: pandas' guess moved 16,223 Mar-26 receipts into other months
    and put 5,396 stock batches' GRN dates after the snapshot they were counted in.
  - Consumption: the consolidated Dec-25..May-26 extract and the monthly ones overlap (the
    monthly Dec, Jan and Mar files are subsets of it) and several monthly extracts book issues
    as negative quantities. Each month comes from exactly one source (a month the consolidated
    extract covers comes from it), and issues are stored as positive quantities.
  - GRN: returns (MvT 122/161, from Jun-26) carry positive quantities and amounts, about
    Rs 1.9 Cr a month; only receipts (101) are kept, the returns are counted.
  - PO: deleted lines (Del Flag 'L') are still in the extract (Aug-26: Rs 34.6 Cr, mostly three
    lines whose quantity is the material code), and Net Price is per 'Price Unit' (10, 15, 30...).
    Lines with a negative quantity (and price) are returns, not orders (Sep-26: whole orders
    at one hospital): an order measure such as fill rate cannot use them, so they are dropped
    and counted, like the GRN's returns.
  - Lines without a material (services, civil works, capital items) are dropped and counted.
    Blank cells are tested as missing, never as the text 'nan': pandas 3 keeps them missing, so
    the old string test dropped nothing.
No document number is kept. The GRN reader never reads GR or PO numbers, so fact_grn has no
order or receipt number column, and fact_po's po_no is an opaque key per purchase order
('PO-0000001'): lines can still be counted per order without SAP's order number leaving the
loader.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from app.core.config import settings
from app.etl.formulary import excel as X
from app.etl.formulary.discover import BuildError, _excel_files
from app.etl.purchase import config as PC
from app.etl.purchase import grn as G

# Source folders, matched on the trimmed folder name anywhere below RAW_DATA_DIR. The GRN
# folder is the purchase build's own (purchase.config.GRN_DIR_RE).
CONSUMPTION_DIR_RE = re.compile(r"^consumption\b", re.I)
INVENTORY_DIR_RE = re.compile(r"^inventory report\b", re.I)
PO_DIR_RE = re.compile(r"^po details\b", re.I)

MONTH_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

# --- Consumption extract (SAP goods movements): output column -> header, matched on
# X.header_key (whitespace collapsed, upper-case: ' Quantity' and '    Quantity' agree).
# Never read: Mat. Doc., Reference, Batch, Customer, User.
CONSUMPTION_COLUMNS = {
    "material": "MATERIAL",
    "material_desc": "MATERIAL DESCRIPTION",
    "plant": "PLNT",
    "sloc": "SLOC",
    "mvt": "MVT",
    "cost_ctr": "COST CTR",
    "qty": "QUANTITY",
    "amount_lc": "AMOUNT IN LC",
    "uom": "EUN",
    "posting_date": "PSTNG DATE",
}
ISSUE_MVT = 201              # goods issue to a cost centre: the only movement fact_consumption holds
FACT_CONSUMPTION_COLUMNS = ["material", "material_desc", "plant", "sloc", "mvt", "cost_ctr", "qty",
                            "amount_lc", "uom", "posting_date", "year", "month_num", "month"]

# --- PO extract (one workbook a month, 43 columns). 'PO No' is read only to key the lines of
# one order (see _order_keys); the number itself never leaves the loader.
PO_COLUMNS = {
    "plant": "PLANT",
    "plant_name": "PLANT NAME",
    "sloc": "STORAGE LOCATION",
    "material": "MATERIAL",
    "material_desc": "MATERIAL DESCRIPTION",
    "vendor_code": "PO VENDOR",
    "vendor_name": "VENDOR NAME",
    "po_ref": "PO NO",
    "doc_type": "DOC.TYPE DESCRIPTION",
    "po_qty": "PO QUANTITY",
    "open_qty": "OPEN QTY",
    "net_price": "NET PRICE",
    "price_unit": "PRICE UNIT",
    "total_value_wo_tax": "TOTAL VALUE WITHOUT TAX",
    # The GST split. `total_value_wo_tax` (the procurement KPI's basis) is EX-tax by
    # definition; without these there was no way to answer a tax question at all.
    "total_value_tax": "TOTAL VALUE WITH TAX",
    "cgst_value": "CGST VALUE",
    "sgst_value": "SGST VALUE",
    "igst_value": "IGST VALUE",
    "major_group": "MAJOR GROUP",
    "minor_group": "MINOR GROUP",
    "po_date": "PO DATE",
    "delivery_date": "DELIVERY DATE",
    "del_flag": "DEL FLAG",
}
PO_OPTIONAL = ("total_value_tax", "cgst_value", "sgst_value", "igst_value", "major_group",
               "minor_group")
PO_NUMBERS = ("po_qty", "open_qty", "net_price", "price_unit", "total_value_wo_tax",
              "total_value_tax", "cgst_value", "sgst_value", "igst_value")
PO_TEXTS = ("plant", "plant_name", "sloc", "material_desc", "vendor_name", "doc_type",
            "major_group", "minor_group", "del_flag")
PO_DATES = ("po_date", "delivery_date")
PO_DELETED_FLAGS = ("L",)    # SAP's deletion indicator; 'S' (blocked) lines are still orders
PO_KEY_FORMAT = "PO-{:07d}"
FACT_PO_COLUMNS = ["plant", "plant_name", "sloc", "material", "material_desc", "vendor_code",
                   "vendor_name", "po_no", "doc_type", "po_qty", "open_qty", "net_price",
                   "total_value_wo_tax", "total_value_tax", "cgst_value", "sgst_value",
                   "igst_value", "major_group", "minor_group", "po_date", "delivery_date",
                   "posting_date", "year", "month_num", "month"]

# --- GRN: the purchase build's reader (app/etl/purchase/grn.py) plus the extras fact_grn keeps.
# TAX, DISCOUNT and MRP — dense (96-99% non-zero): "what did we pay in GST", "which items carry
# a discount", "what is our margin at receipt" all failed until these were read. Deliberately
# NOT kept: Selling Price and Free Qty (100% filled, 100% ZERO), Formulary (SAP's own flag,
# messy; the real one is on dim_material and in the Item Masters), and PR No / PR Date / PO TAT
# (4-6% filled — a requisition-cycle answer from 6% of rows would mislead).
GRN_EXTRAS = ("po_date", "expiry_date", "po_to_gr_tat", "pr_to_gr_tat", "tax_amount",
              "total_with_tax", "total_mrp_value")
GRN_RENAME = {"total_wo_tax": "total_amount_wo_tax", "total_with_tax": "total_amount_with_tax"}
GRN_DATES = ("po_date", "gr_date", "expiry_date")
FACT_GRN_COLUMNS = ["plant", "sloc", "material", "material_desc", "vendor_code", "vendor_name",
                    "gr_qty", "base_price", "net_price", "unit_mrp",
                    "total_amount_wo_tax", "tax_pct", "tax_amount", "total_amount_with_tax",
                    "discount_value", "total_mrp_value", "po_to_gr_tat", "pr_to_gr_tat",
                    "major_group", "minor_group", "po_date", "gr_date", "expiry_date",
                    "posting_date", "year", "month_num", "month"]

Log = Callable[[str], None]


@dataclass
class SourceRead:
    """One source's curated frame and what the loader read and left out. The report holds
    counts, values and file names only: run_etl writes it to curated/_dq_report.json."""
    frame: pd.DataFrame
    report: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Folders, sheets and cells
# --------------------------------------------------------------------------- #
def raw_dir(d=None) -> Path:
    return Path(d or settings.RAW_DATA_DIR).expanduser()


def find_source_dir(pattern: re.Pattern, what: str, raw=None) -> Path:
    """The one folder below RAW_DATA_DIR whose name matches `pattern`."""
    root = raw_dir(raw)
    if not root.is_dir():
        raise BuildError(f"RAW_DATA_DIR does not exist: {root}")
    found = sorted(d for d in root.rglob("*") if d.is_dir() and pattern.match(d.name.strip()))
    if not found:
        raise BuildError(f"no {what} folder under {root}")
    if len(found) > 1:
        raise BuildError(f"more than one {what} folder: " + ", ".join(repr(str(d)) for d in found))
    return found[0]


def _workbooks(folder: Path) -> list:
    files = _excel_files(folder)
    if not files:
        raise BuildError(f"no Excel workbooks in {folder}")
    return files


def _blank(v) -> bool:
    return X.text(v) is None


def _find_header(wb: X.Workbook, sheet: str, need: set) -> tuple:
    """(row index, normalised header, set()) of the first row within X.HEADER_SCAN_ROWS that
    holds every needed header, else (None, [], the fewest headers any of those rows missed)."""
    closest = set(need)
    for i, row in enumerate(wb.rows(sheet)):
        if i >= X.HEADER_SCAN_ROWS:
            break
        hdr = [X.header_key(v) for v in (row or ())]
        missing = need - set(hdr)
        if not missing:
            return i, hdr, set()
        if len(missing) < len(closest):
            closest = missing
    return None, [], closest


def _sheet_is_empty(wb: X.Workbook, sheet: str) -> bool:
    return all(_blank(v) for row in wb.rows(sheet) for v in (row or ()))


def _read_columns(wb: X.Workbook, sheet: str, header_row: int, hdr: list, columns: dict) -> tuple:
    """({output column: raw cells}, lines) below the header row. Fully empty lines are skipped;
    a column whose header is absent comes back all None."""
    pos = {c: hdr.index(h) for c, h in columns.items() if h in hdr}
    width = max(pos.values()) + 1
    buf = []
    it = wb.rows(sheet)
    for _ in range(header_row + 1):
        next(it)
    for r in it:
        if r is None:
            continue
        if len(r) < width:
            r = list(r) + [None] * (width - len(r))
        vals = tuple(r[i] for i in pos.values())
        if all(_blank(v) for v in vals):
            continue
        buf.append(vals)
    n = len(buf)
    got = dict(zip(pos, zip(*buf))) if buf else {c: () for c in pos}
    return {c: list(got[c]) if c in got else [None] * n for c in columns}, n


def _convert(values: Sequence, fn: Callable) -> list:
    """fn over a column, each distinct value converted once (codes and names repeat a lot)."""
    memo: dict = {}
    out = []
    for v in values:
        try:
            out.append(memo[v] if v in memo else memo.setdefault(v, fn(v)))
        except TypeError:                       # unhashable: convert without the memo
            out.append(fn(v))
    return out


def _texts(values) -> pd.Series:
    return pd.Series(_convert(values, X.text), dtype="str")


def _codes(values) -> pd.Series:
    return pd.Series(_convert(values, G.clean_code), dtype="str")


def _numbers(values) -> pd.Series:
    return pd.Series(_convert(values, X.number), dtype="float64")


def _dates(values) -> tuple:
    """(datetime64[us] Series, the parse counts): day-first, through the GRN reader's parser."""
    parsed = G.parse_gr_dates(values)
    stamps = pd.to_datetime(pd.Series(parsed.dates, dtype="object"))
    return stamps.astype("datetime64[us]"), parsed


def _date_report(parsed) -> dict:
    return {**parsed.kinds, "blank": parsed.blank, "unparseable": parsed.unparseable}


def _periods(dates: pd.Series) -> pd.Series:
    return dates.dt.strftime("%Y-%m")


def _month_names(month_num: pd.Series) -> pd.Series:
    return month_num.map(lambda m: MONTH_NAME.get(int(m)) if pd.notna(m) else None)


def _add_period(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df[date_col]
    df["posting_date"] = d
    df["year"] = d.dt.year.astype("Int64")
    df["month_num"] = d.dt.month.astype("Int64")
    df["month"] = _month_names(df["month_num"])
    return df


def _money(v) -> float:
    return round(float(v), 2)


# --------------------------------------------------------------------------- #
# Consumption
# --------------------------------------------------------------------------- #
@dataclass
class ConsumptionFile:
    """One consumption workbook: its dated lines, signs as booked, and the months they fall in."""
    file: str
    sheet: str
    lines: pd.DataFrame
    periods: dict                    # 'YYYY-MM' -> lines
    undated: int                     # lines without a posting date (SAP's sum line at the end)
    dates: dict

    @property
    def consolidated(self) -> bool:
        return len(self.periods) > 1


def _read_consumption_file(path: Path, log: Log) -> Optional[ConsumptionFile]:
    """One consumption workbook -> its dated lines; None for an empty workbook."""
    need = set(CONSUMPTION_COLUMNS.values())
    with X.Workbook(path) as wb:
        found, closest = [], (None, need)
        for sh in wb.sheet_names:
            row, hdr, missing = _find_header(wb, sh, need)
            if row is not None:
                found.append((sh, row, hdr))
            elif len(missing) < len(closest[1]):
                closest = (sh, missing)
        if not found:
            if all(_sheet_is_empty(wb, sh) for sh in wb.sheet_names):
                return None
            raise BuildError(f"{path.name}: no sheet has the consumption columns (closest: "
                             f"{closest[0]!r} lacks {', '.join(sorted(closest[1]))})")
        if len(found) > 1:
            raise BuildError(f"{path.name}: sheets {[s for s, _, _ in found]} all hold "
                             f"consumption lines; expected one")
        sheet, row, hdr = found[0]
        cols, n = _read_columns(wb, sheet, row, hdr, CONSUMPTION_COLUMNS)
    dates, parsed = _dates(cols["posting_date"])
    df = pd.DataFrame({
        "material": _codes(cols["material"]),
        "material_desc": _texts(cols["material_desc"]),
        "plant": _texts(cols["plant"]),
        "sloc": _texts(cols["sloc"]),
        "mvt": _numbers(cols["mvt"]),
        "cost_ctr": _codes(cols["cost_ctr"]),
        "qty": _numbers(cols["qty"]),
        "amount_lc": _numbers(cols["amount_lc"]),
        "uom": _texts(cols["uom"]),
        "posting_date": dates,
    })
    dated = df["posting_date"].notna()
    df = df[dated].reset_index(drop=True)
    if df.empty:
        raise BuildError(f"{path.name}: {n:,} lines but no readable posting date")
    periods = dict(sorted(Counter(_periods(df["posting_date"])).items()))
    log(f"[ingest] consumption {path.name}: {len(df):,} dated lines in {', '.join(periods)}")
    return ConsumptionFile(path.name, sheet, df, periods, int((~dated).sum()),
                           _date_report(parsed))


def choose_consumption_sources(files: Sequence[ConsumptionFile]) -> dict:
    """period -> the one workbook the month is taken from. A month a consolidated (multi-month)
    workbook covers comes from it; any other month from its monthly workbook. Two consolidated
    workbooks, or two monthly ones, for the same month abort: either way it would be counted
    twice."""
    chosen, errors = {}, []
    for period in sorted({p for f in files for p in f.periods}):
        wide = [f for f in files if f.consolidated and period in f.periods]
        narrow = [f for f in files if not f.consolidated and period in f.periods]
        pick = wide or narrow
        if len(pick) > 1:
            errors.append(f"{period} is in {', '.join(repr(f.file) for f in pick)}")
        else:
            chosen[period] = pick[0]
    if errors:
        raise BuildError("consumption: each month needs exactly one source:\n  "
                         + "\n  ".join(errors))
    return chosen


def issue_sign(qty: pd.Series, where: str) -> int:
    """+1 when a source books issues as positive quantities, -1 when as negative (several
    monthly extracts do). A source booking both ways cannot tell an issue from a reversal."""
    pos, neg = int((qty > 0).sum()), int((qty < 0).sum())
    if pos and neg:
        raise BuildError(f"{where}: MvT {ISSUE_MVT} quantities are positive on {pos:,} lines and "
                         f"negative on {neg:,}; issues must be booked one way")
    return -1 if neg else 1


def load_consumption(raw=None, log: Log = print) -> SourceRead:
    """Stores issues (MvT 201), one source per month, quantities and amounts positive. Lines of
    other movement types and lines without a quantity are dropped and counted."""
    folder = find_source_dir(CONSUMPTION_DIR_RE, "'Consumption ...'", raw)
    files, skipped = [], []
    for p in _workbooks(folder):
        f = _read_consumption_file(p, log)
        if f is None:
            skipped.append({"file": p.name, "reason": "empty workbook"})
        else:
            files.append(f)
    if not files:
        raise BuildError(f"no consumption lines in {folder}")
    chosen = choose_consumption_sources(files)

    sources: list = []                                  # each source once, by its first month
    for src in chosen.values():
        if all(src is not s for s in sources):
            sources.append(src)
    frames, months = [], []
    for f in sources:
        mine = [p for p, src in chosen.items() if src is f]
        period = _periods(f.lines["posting_date"])
        lines = f.lines[period.isin(mine)].copy()         # the workbook's own line order
        issue = lines["mvt"] == ISSUE_MVT
        signs = {p: issue_sign(lines.loc[issue & (period == p), "qty"], f"{f.file} {p}")
                 for p in mine}
        sign = period[lines.index].map(signs)
        lines["qty"] = lines["qty"] * sign
        lines["amount_lc"] = lines["amount_lc"] * sign
        positive = lines["qty"].fillna(0) > 0
        kept = lines[issue & positive]
        frames.append(kept)
        for p in mine:
            on = period[lines.index] == p
            got = kept[on[kept.index]]
            months.append({
                "period": p, "file": f.file,
                "source": "consolidated" if f.consolidated else "monthly",
                "issues_booked": "negative (signs flipped)" if signs[p] < 0 else "positive",
                "lines_read": int(on.sum()), "other_movements": int((on & ~issue).sum()),
                "no_quantity": int((on & issue & ~positive).sum()), "lines": len(got),
                "qty": round(float(got["qty"].sum()), 3), "amount": _money(got["amount_lc"].sum()),
                "plants": int(got["plant"].nunique()),
            })
    months.sort(key=lambda m: m["period"])
    out = pd.concat(frames, ignore_index=True)
    out = _add_period(out, "posting_date")[FACT_CONSUMPTION_COLUMNS]
    used = {f.file for f in sources}
    report = {
        "folder": folder.name,
        "months": months,
        "superseded": [{"file": f.file, "periods": list(f.periods)} for f in files
                       if f.file not in used],
        "files": [{"file": f.file, "sheet": f.sheet, "periods": f.periods, "undated": f.undated,
                   "dates": f.dates} for f in files],
        "skipped": skipped,
    }
    log(f"[ingest] consumption: {len(out):,} issue lines; "
        + ", ".join(f"{m['period']} from {m['file']}" for m in months))
    return SourceRead(out, report)


# --------------------------------------------------------------------------- #
# Inventory snapshot
# --------------------------------------------------------------------------- #
def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_id(s: pd.Series) -> pd.Series:
    """Normalize numeric-ID columns: floats like '100301.0' -> '100301'.

    SAP codes are read inconsistently as int/float/str across files; without this
    a float-read code ('100301.0') won't join to an int-read one ('100301')."""
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def load_inventory(raw=None, log: Log = print) -> SourceRead:
    """The stock snapshot: the one workbook in the inventory folder (a second one would make
    'the' snapshot ambiguous)."""
    folder = find_source_dir(INVENTORY_DIR_RE, "'Inventory report ...'", raw)
    files = _workbooks(folder)
    if len(files) > 1:
        raise BuildError(f"{folder.name}: expected one inventory snapshot, found "
                         + ", ".join(repr(p.name) for p in files))
    df = pd.read_excel(files[0], engine="openpyxl")
    df = _norm_headers(df)

    def g(name, *alts):
        for n in (name, *alts):
            if n in df.columns:
                return df[n]
        return pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "plant": g("Plant").astype(str).str.strip(),
        "sloc": g("Storage Location").astype(str).str.strip(),
        "sloc_desc": g("Storage Location Description").astype(str).str.strip(),
        "material": _clean_id(g("Material")),
        "material_desc": g("Material Desc").astype(str).str.strip(),
        "hsn": g("HSN Code").astype(str).str.strip(),
        "batch": g("Batch").astype(str).str.strip(),
        "qty": _to_num(g("    Quantity", "Quantity")),
        "uom": g("Basic UOM").astype(str).str.strip(),
        "total_cost": _to_num(g("   Total Cost", "Total Cost")),
        "total_mrp": _to_num(g("    Total MRP", "Total MRP")),
        "moving_avg_price": _to_num(g("Moving Average Price(Unit)")),
        "aging_days": _to_num(g(" Aging", "Aging")),
        "manufacturer_desc": g("Manufacturer Description").astype(str).str.strip(),
        "generic_name": g("Generic Name").astype(str).str.strip(),
        "formulary": g("Formulary").astype(str).str.strip(),
        "material_type": g("Material Type").astype(str).str.strip(),
        "material_group": g("Material Group").astype(str).str.strip(),
        "major_group_desc": g("Major Group Description").astype(str).str.strip(),
        "minor_group_desc": g("Minor Group Description").astype(str).str.strip(),
    })
    # Each date column holds real dates on some rows and 'dd.mm.yyyy' text on others (43,763 of
    # 97,590 GRN dates): read day-first, as everywhere else in this module.
    dates = {}
    for col, header in (("expiry_date", "Expiry Date"), ("grn_date", "GRN Date"),
                        ("snapshot_date", "Date")):
        out[col], parsed = _dates(g(header).tolist())
        dates[col] = _date_report(parsed)
    snapshot = out["snapshot_date"].max()
    log(f"[ingest] inventory: {len(out):,} rows, snapshot={snapshot}")
    return SourceRead(out, {"folder": folder.name, "file": files[0].name, "rows": len(out),
                            "snapshot": None if pd.isna(snapshot) else str(snapshot.date()),
                            "dates": dates})


# --------------------------------------------------------------------------- #
# GRN
# --------------------------------------------------------------------------- #
def load_grn(raw=None, cache=None, log: Log = print) -> SourceRead:
    """Goods receipts (MvT 101) of every GRN workbook, through the purchase build's reader:
    day-first dates, one month per workbook, lines without a material dropped, no document
    number read. Returns and any other movement are dropped and counted per workbook."""
    read = G.read_grn_files(raw=raw_dir(raw), cache=cache, extra=GRN_EXTRAS, log=log)
    df = read.frame
    receipt = df["mvt_type"].eq(PC.RECEIPT_MVT).fillna(False).astype(bool)
    others = (df.loc[~receipt, ["period", "total_wo_tax"]]
              .assign(mvt_type=df.loc[~receipt, "mvt_type"].fillna(""))
              .groupby(["period", "mvt_type"])["total_wo_tax"].agg(["size", "sum"]))
    files = []
    for rep in read.files:
        kept = df[receipt & (df["period"] == rep.period)]
        files.append({
            "file": rep.file, "sheet": rep.sheet, "period": rep.period, "lines_read": rep.rows,
            "no_material": {"lines": rep.blank_material_lines,
                            "value": _money(sum(v["value"] for v in rep.blank_material.values()))},
            "not_receipts": {m: {"lines": int(r["size"]), "value": _money(r["sum"])}
                             for (p, m), r in others.iterrows() if p == rep.period},
            "lines": len(kept), "qty": round(float(kept["gr_qty"].sum()), 3),
            "value": _money(kept["total_wo_tax"].sum()), "gr_dates": rep.dates,
        })
    g = df[receipt].rename(columns=GRN_RENAME).reset_index(drop=True)
    for c in GRN_DATES:
        g[c] = g[c].astype("datetime64[us]")
    g["posting_date"] = g["gr_date"]
    # The month is the workbook's: every readable GR date in it falls in that month, and a line
    # whose date could not be read (NaT) still belongs to it.
    g["year"] = g["period"].str[:4].astype("Int64")
    g["month_num"] = g["period"].str[5:7].astype("Int64")
    g["month"] = _month_names(g["month_num"])
    g = g[FACT_GRN_COLUMNS]
    log(f"[ingest] grn: {len(g):,} receipt lines from {len(read.files)} files "
        f"({', '.join(read.periods)})")
    return SourceRead(g, {"folder": read.folder.name if read.folder else None, "files": files,
                          "warnings": list(read.warnings)})


# --------------------------------------------------------------------------- #
# PO
# --------------------------------------------------------------------------- #
@dataclass
class PoFile:
    """One PO workbook: its kept lines (po_ref still the SAP order number, in memory only), the
    months of their PO dates and what was left out."""
    file: str
    lines: pd.DataFrame
    periods: dict                    # 'YYYY-MM' -> kept lines
    report: dict


def read_po_sheet(wb: X.Workbook, sheet: str) -> Optional[tuple]:
    """(lines, lines read, date counts) of one sheet in the PO layout, every line as booked
    (deleted ones and those without a material included; po_ref is SAP's order number, for
    keying only). None for an empty sheet; BuildError for a sheet with data but no PO header."""
    need = {h for c, h in PO_COLUMNS.items() if c not in PO_OPTIONAL}
    row, hdr, missing = _find_header(wb, sheet, need)
    if row is None:
        if _sheet_is_empty(wb, sheet):
            return None
        raise BuildError(f"{wb.path.name} sheet {sheet!r}: no PO header (lacks "
                         f"{', '.join(sorted(missing))})")
    cols, n = _read_columns(wb, sheet, row, hdr, PO_COLUMNS)
    df = pd.DataFrame({c: _numbers(cols[c]) for c in PO_NUMBERS})
    for c in ("material", "vendor_code", "po_ref"):
        df[c] = _codes(cols[c])
    for c in PO_TEXTS:
        df[c] = _texts(cols[c])
    dates = {}
    for c in PO_DATES:
        df[c], parsed = _dates(cols[c])
        dates[c] = _date_report(parsed)
    return df, n, dates


def po_deleted(lines: pd.DataFrame) -> pd.Series:
    """Lines carrying SAP's deletion indicator."""
    return lines["del_flag"].str.upper().isin(PO_DELETED_FLAGS).fillna(False).astype(bool)


def po_returns(lines: pd.DataFrame) -> pd.Series:
    """Return lines: a negative order quantity. They are not orders (their 'open' quantity
    would count as goods still to come), so an order measure leaves them out."""
    return (lines["po_qty"] < 0).fillna(False).astype(bool)


def _read_po_file(path: Path, log: Log) -> Optional[PoFile]:
    """One PO workbook -> its order lines, deleted lines, lines without a material and return
    lines left out and counted; None for an empty workbook (HCG's folder holds an empty
    Book.xlsx)."""
    with X.Workbook(path) as wb:
        # SHEET 0 ONLY — and that is correct, however wrong it looks.
        #
        # Six of these workbooks carry a second sheet ("Capex" Dec-25, "Capex PO" Jan-26,
        # "Dom Capital PO" Feb-26, "Capex deatils" Jul-26, "Capital PO" Aug-26, "Capex"
        # Sep-26: 1,110 lines, Rs 158.48 Cr) in the identical 43-column schema, which reads
        # exactly like data being dropped on the floor. It is not. Every one of those rows has
        # an EXACT twin in the main sheet — same PO, material, quantity and value — and every
        # PO on a tab is on the main sheet too. The tabs are a convenience view the analyst
        # made of capital POs that are already in the extract; reading them would double-count
        # every capital line. (July's third sheet, "GL details", holds no order lines at all.)
        # tests/test_source_completeness.py pins this so the next person who notices the
        # unread tab does not "fix" it either.
        sheet = wb.sheet_names[0]
        got = read_po_sheet(wb, sheet)
        others = wb.sheet_names[1:]
    if got is None:
        return None
    df, n, dates = got
    value = df["total_value_wo_tax"].fillna(0.0)
    deleted = po_deleted(df)
    no_material = df["material"].isna() & ~deleted
    returned = po_returns(df) & ~deleted & ~no_material
    kept = df[~deleted & ~no_material & ~returned].copy()
    # Net Price is the price of 'Price Unit' units (Total value = PO Quantity x Net Price /
    # Price Unit), so the unit price divides by it.
    per = kept["price_unit"]
    scaled = per.notna() & (per > 0) & (per != 1)
    kept.loc[scaled, "net_price"] = kept.loc[scaled, "net_price"] / per[scaled]
    # A quantity equal to the material code is a keying slip (Aug-26: three lines, Rs 33.47 Cr,
    # all deleted); counted so one that is not deleted shows up in the report.
    qty_is_code = int((kept["po_qty"] == pd.to_numeric(kept["material"], errors="coerce")).sum())
    periods = dict(sorted(Counter(_periods(kept["po_date"].dropna())).items()))
    if not periods:
        raise BuildError(f"{path.name}: no order line with a readable PO date")
    blank = df[no_material]
    report = {
        "file": path.name, "sheet": sheet, "periods": periods, "lines_read": n,
        "deleted": {"lines": int(deleted.sum()), "value": _money(value[deleted].sum())},
        "no_material": {"lines": len(blank), "value": _money(value[no_material].sum()),
                        "by_doc_type": {k: _money(v) for k, v in blank.groupby(
                            blank["doc_type"].fillna(""))["total_value_wo_tax"].sum().items()}},
        "returns": {"lines": int(returned.sum()), "value": _money(value[returned].sum())},
        "price_unit_scaled": int(scaled.sum()), "qty_equals_material_code": qty_is_code,
        "lines": len(kept), "value": _money(kept["total_value_wo_tax"].sum()),
        "dates": dates, "sheets_not_read": others,
    }
    log(f"[ingest] po {path.name}: {len(kept):,} of {n:,} lines kept "
        f"({report['deleted']['lines']:,} deleted, {len(blank):,} without a material, "
        f"{report['returns']['lines']:,} returns)")
    return PoFile(path.name, kept, periods, report)


def order_keys(refs: pd.Series) -> pd.Series:
    """SAP order numbers -> opaque keys 'PO-0000001', numbered in order of first appearance.
    Lines of one order share a key; the number itself is not kept."""
    codes, _ = pd.factorize(refs)
    return pd.Series([PO_KEY_FORMAT.format(c + 1) if c >= 0 else None for c in codes],
                     index=refs.index, dtype="str")


def load_po(raw=None, log: Log = print) -> SourceRead:
    """Purchase-order lines of every PO workbook (its first sheet). Deleted lines, lines
    without a material and return lines are dropped and counted, Net Price is per unit, and
    the order number becomes an opaque per-order key. Two workbooks holding orders of the same
    month abort (a re-extract beside the original would count the month twice)."""
    folder = find_source_dir(PO_DIR_RE, "'PO details ...'", raw)
    reads, skipped = [], []
    for p in _workbooks(folder):
        r = _read_po_file(p, log)
        if r is None:
            skipped.append({"file": p.name, "reason": "empty workbook"})
        else:
            reads.append(r)
    if not reads:
        raise BuildError(f"no PO lines in {folder}")
    seen, errors = {}, []
    for r in reads:
        for period in r.periods:
            if period in seen:
                errors.append(f"{period} is in {seen[period]!r} and {r.file!r}")
            seen.setdefault(period, r.file)
    if errors:
        raise BuildError("PO: each month needs exactly one workbook:\n  " + "\n  ".join(errors))
    reads.sort(key=lambda r: min(r.periods))
    df = pd.concat([r.lines for r in reads], ignore_index=True)
    df["po_no"] = order_keys(df.pop("po_ref"))
    out = _add_period(df, "po_date")[FACT_PO_COLUMNS]
    log(f"[ingest] po: {len(out):,} lines from {len(reads)} files ({', '.join(sorted(seen))})")
    return SourceRead(out, {"folder": folder.name, "files": [r.report for r in reads],
                            "skipped": skipped})
