"""HCG's SAP GRN (goods receipt) extracts -> one clean line-level frame: the one reader the
purchase build and the legacy curated import (app/etl/ingest.py) share.

  find_grn_dir, grn_files  the GRN folder under FORMULARY_RAW_DIR and its workbooks
  ensure_cached            one workbook into the cache (read once) -> its GrnFileReport
  cached_report            a workbook's cached report, if the cache holds it (no read)
  read_grn_file            one workbook -> (lines, GrnFileReport), through the cache
  read_grn_files           every workbook, each assigned to its month -> GrnRead
  apply_scope              the analysis scope (config.Scope) -> (kept lines, funnel)
  parse_gr_date(s)         day-first dates from date cells, Excel serials and text
  single_period            the one 'YYYY-MM' a set of dates falls in, else BuildError
  clean_code(s)            SAP codes as clean text ('100301.0' -> '100301')

What the extracts get wrong, and what the reader does about it:
  - Header spellings vary ('    Base Price' vs '     Base Price', 'mvt type', an unnamed
    column in Dec-25): headers are matched whitespace-normalised and upper-case, and only
    the needed columns are read.
  - GR Date is a real date in some files, 'dd.mm.yyyy' text in others (Mar-26, Jun-26) and
    both in Aug-26. Text is read DAY-FIRST, explicitly: pandas' guess put 16,223 Mar-26
    lines into other months. A date that cannot be read is counted, never guessed.
  - Each workbook holds one calendar month, whatever its name says (the folder is still
    'GRN Dec 25 to May 2026'): the month comes from the GR dates, and a workbook spanning
    two months, or two workbooks for one month, abort.
  - Lines without a material (84 in Jun-26) are dropped and counted per movement type, so
    the funnel still reproduces HCG's sheet, whose movement-type step counts them.
  - Returns (movement types 122, 161; none before Jun-26) are kept and counted: the scope,
    not the reader, decides what an analysis counts.

No identifier is read: GR, PO, PR, invoice and reference numbers never leave the workbook.
Each workbook's lines are cached as parquet (+ a .json report) in PURCHASE_CACHE_DIR, keyed
on its path, size, mtime and the reader version: reading the nine workbooks takes minutes
once and about a second from the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.etl.formulary import excel as X
from app.etl.formulary.classify import norm_code
from app.etl.formulary.discover import BuildError, _excel_files
from app.etl.purchase import config as C

# Every column the cache holds; a read returns the core columns plus the extras asked for.
ALL_COLUMNS = {**C.GRN_COLUMNS, **C.GRN_EXTRA_COLUMNS}
_UPPER = ("plant", "sloc")
_CODES = ("material", "vendor_code", "mvt_type")
SCHEMA = pa.schema(
    [("period", pa.string())]
    + [(c, pa.date32() if c in C.GRN_DATE_COLUMNS else
        pa.float64() if c in C.GRN_NUMBER_COLUMNS else pa.string()) for c in ALL_COLUMNS])

# Text dates: day first, one separator, a four-digit year, an optional time ('04.08.2026',
# '4-8-2026', '04/08/2026 00:00'); ISO 'yyyy-mm-dd' is unambiguous and also accepted.
_DMY = re.compile(r"^(\d{1,2})([./-])(\d{1,2})\2(\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$")
# An Excel serial is only read as a date inside this window; any other number in a date
# column is something else and counted as unreadable.
SERIAL_WINDOW = (date(1990, 1, 1), date(2100, 12, 31))


# --------------------------------------------------------------------------- #
# Cell values
# --------------------------------------------------------------------------- #
def _missing(v) -> bool:
    if v is None or v is pd.NaT or v is pd.NA:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and not v.strip()


def clean_code(v) -> Optional[str]:
    """A SAP code as text: 100301.0, '100301.0', ' 100301 ' -> '100301'; None for blanks,
    NaN / NA and Excel error cells (never the string 'nan')."""
    return None if _missing(v) else norm_code(v)


def clean_codes(s: pd.Series) -> pd.Series:
    """clean_code over a column; missing stays missing (pandas 'str' dtype)."""
    return pd.Series([clean_code(v) for v in s], index=s.index, dtype="str")


def _date_kind(v) -> str:
    if isinstance(v, (datetime, date)):
        return "date"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "serial"
    return "text"


def parse_gr_date(v) -> Optional[date]:
    """One GRN date cell -> date, or None when it is blank or cannot be read.

    datetime / date cells as they are; Excel serial numbers (pyxlsb never converts) inside
    SERIAL_WINDOW; text 'dd.mm.yyyy', 'dd-mm-yyyy', 'dd/mm/yyyy' DAY-FIRST ('04.08.2026' is
    4 August) and ISO 'yyyy-mm-dd'. Nothing else is guessed: '08.13.2026' (no 13th month),
    '31.02.2026' and two-digit years are None."""
    if _missing(v) or isinstance(v, bool):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)):
        try:
            d = (X.EXCEL_BASE + timedelta(days=float(v))).date()
        except (OverflowError, ValueError):
            return None
        return d if SERIAL_WINDOW[0] <= d <= SERIAL_WINDOW[1] else None
    s = str(v).strip()
    m = _DMY.match(s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(3)), int(m.group(4))
    else:
        m = _ISO.match(s)
        if not m:
            return None
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


class ParsedDates(NamedTuple):
    dates: list            # date or None, one per value
    blank: int             # empty cells
    unparseable: int       # non-empty cells that are not a readable date
    kinds: dict            # non-empty cells by kind: date / serial / text


def parse_gr_dates(values: Iterable) -> ParsedDates:
    """parse_gr_date over a column, with the counts a read report needs. Each distinct value
    is parsed once (a month of lines holds ~30 distinct dates)."""
    values = list(values)
    memo: dict = {}
    out, blank, bad, kinds = [], 0, 0, Counter()
    for v in values:
        if _missing(v):
            out.append(None)
            blank += 1
            continue
        kinds[_date_kind(v)] += 1
        try:
            d = memo[v] if v in memo else memo.setdefault(v, parse_gr_date(v))
        except TypeError:                       # unhashable: parse without the memo
            d = parse_gr_date(v)
        bad += d is None
        out.append(d)
    return ParsedDates(out, blank, bad, dict(sorted(kinds.items())))


def single_period(dates: Iterable, where: str) -> str:
    """The one 'YYYY-MM' all readable dates fall in. BuildError when there is none or more
    than one: a GRN workbook (or HCG worksheet) holds exactly one month."""
    months = Counter(f"{d.year:04d}-{d.month:02d}" for d in dates
                     if isinstance(d, date) and not _missing(d))
    if not months:
        raise BuildError(f"{where}: no readable GR date, so its month is unknown")
    if len(months) > 1:
        spread = ", ".join(f"{p}: {n:,}" for p, n in sorted(months.items()))
        raise BuildError(f"{where}: GR dates span {len(months)} months ({spread}); each "
                         f"workbook must hold exactly one month")
    return next(iter(months))


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def find_grn_dir(raw=None) -> Path:
    """The one 'GRN ...' folder under FORMULARY_RAW_DIR."""
    root = C.raw_dir(raw)
    if not root.is_dir():
        raise BuildError(f"FORMULARY_RAW_DIR does not exist: {root}")
    found = sorted(d for d in root.rglob("*") if d.is_dir() and C.GRN_DIR_RE.match(d.name.strip()))
    if not found:
        raise BuildError(f"no 'GRN ...' folder under {root}")
    if len(found) > 1:
        raise BuildError("more than one GRN folder: " + ", ".join(repr(str(d)) for d in found))
    return found[0]


def grn_files(raw=None) -> list:
    return _excel_files(find_grn_dir(raw))


@dataclass
class GrnFileReport:
    """What one GRN workbook held. Counts only: no line, no identifier."""
    file: str
    sheet: str
    period: str
    rows: int                          # lines read (fully empty rows not counted)
    lines: int                         # lines kept: rows - lines without a material
    blank_rows: int
    blank_material: dict               # mvt type -> {"lines", "value"} of the dropped lines
    mvt: dict                          # mvt type -> lines, every line read
    returns: int                       # lines of config.RETURN_MVTS (kept)
    dates: dict                        # GR Date cells: date / serial / text / blank / unparseable
    value: float                       # Total Amount without tax of the kept lines
    qty: float                         # GR Qty of the kept lines
    source_size: int
    source_mtime_ns: int
    cache_file: str = ""
    cached: bool = False               # served from the cache on this read
    secs: float = 0.0

    @property
    def blank_material_lines(self) -> int:
        return sum(v["lines"] for v in self.blank_material.values())

    @property
    def unparseable_dates(self) -> int:
        return self.dates.get("unparseable", 0)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GrnRead:
    """Every GRN workbook read: the lines of the chosen months and a report per file."""
    folder: Optional[Path]
    frame: pd.DataFrame
    files: list = field(default_factory=list)          # GrnFileReport, by period
    warnings: list = field(default_factory=list)

    @property
    def periods(self) -> list:
        return sorted({f.period for f in self.files})

    def blank_lines(self) -> pd.DataFrame:
        """The lines dropped for having no material: period, mvt_type, lines, value. Passed
        to apply_scope so the funnel's first steps count them, like HCG's sheet."""
        rows = [{"period": f.period, "mvt_type": m, "lines": v["lines"], "value": v["value"]}
                for f in self.files for m, v in f.blank_material.items()]
        return pd.DataFrame(rows, columns=["period", "mvt_type", "lines", "value"])

    def report(self) -> dict:
        return {"folder": self.folder.name if self.folder else None,
                "files": [f.as_dict() for f in self.files], "warnings": list(self.warnings)}


def _cache_key(path: Path, st: os.stat_result) -> str:
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|v{C.GRN_READER_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _cache_paths(cache, key: str) -> tuple:
    d = C.cache_dir(cache)
    return d / f"grn_{key}.parquet", d / f"grn_{key}.json"


def _choose_sheet(wb: X.Workbook) -> tuple:
    """(sheet, header row index, normalised header) of the one sheet holding GRN lines."""
    need = set(C.GRN_COLUMNS.values())
    found, closest = [], (None, need)
    for sh in wb.sheet_names:
        for i, row in enumerate(wb.rows(sh)):
            if i >= X.HEADER_SCAN_ROWS:
                break
            hdr = [X.header_key(v) for v in (row or ())]
            missing = need - set(hdr)
            if not missing:
                found.append((sh, i, hdr))
                break
            if len(missing) < len(closest[1]):
                closest = (sh, missing)
    if not found:
        raise BuildError(f"{wb.path.name}: no sheet has the GRN columns (closest: "
                         f"{closest[0]!r} lacks {', '.join(sorted(closest[1]))})")
    if len(found) > 1:
        raise BuildError(f"{wb.path.name}: sheets {[s for s, _, _ in found]} all hold GRN lines; "
                         f"expected one")
    return found[0]


def _convert(col: str, vals: tuple) -> list:
    if col in C.GRN_NUMBER_COLUMNS:
        fn = X.number
    elif col in _CODES:
        fn = clean_code
    elif col in _UPPER:
        fn = X.upper
    elif col == "manufacturer":
        fn = X.name
    else:
        fn = X.text
    memo = {}
    out = []
    for v in vals:
        try:
            out.append(memo[v] if v in memo else memo.setdefault(v, fn(v)))
        except TypeError:
            out.append(fn(v))
    return out


def _read_workbook(path: Path, log=print) -> tuple:
    """(pyarrow table of the lines with a material, report fields) of one GRN workbook."""
    t0 = time.time()
    with X.Workbook(path) as wb:
        sheet, hdr_row, hdr = _choose_sheet(wb)
        pos = {c: hdr.index(h) for c, h in ALL_COLUMNS.items() if h in hdr}
        read = list(pos)
        width = max(pos.values()) + 1
        buf, blank_rows = [], 0
        it = wb.rows(sheet)
        for _ in range(hdr_row + 1):
            next(it)
        for r in it:
            if r is None:
                blank_rows += 1
                continue
            if len(r) < width:
                r = list(r) + [None] * (width - len(r))
            vals = tuple(r[pos[c]] for c in read)
            if all(_missing(v) for v in vals):
                blank_rows += 1
                continue
            buf.append(vals)
    if not buf:
        raise BuildError(f"{path.name} sheet {sheet!r} read 0 lines — a reader failure, not an "
                         f"empty month")
    n = len(buf)
    cols = dict(zip(read, zip(*buf)))
    buf.clear()
    data, dates = {}, {}
    for c in ALL_COLUMNS:
        if c not in cols:
            data[c] = [None] * n
        elif c in C.GRN_DATE_COLUMNS:
            parsed = parse_gr_dates(cols[c])
            data[c] = parsed.dates
            if c == "gr_date":
                dates = {**parsed.kinds, "blank": parsed.blank, "unparseable": parsed.unparseable}
        else:
            data[c] = _convert(c, cols[c])
    period = single_period(data["gr_date"], path.name)
    keep = [m is not None for m in data["material"]]
    mvt, blank = Counter(data["mvt_type"]), {}
    for m, v, k in zip(data["mvt_type"], data["total_wo_tax"], keep):
        if not k:
            b = blank.setdefault(m or "", {"lines": 0, "value": 0.0})
            b["lines"] += 1
            b["value"] += v or 0.0
    table = pa.table({"period": [period] * n, **data}, schema=SCHEMA).filter(pa.array(keep))
    value = sum(v for v, k in zip(data["total_wo_tax"], keep) if k and v is not None)
    qty = sum(v for v, k in zip(data["gr_qty"], keep) if k and v is not None)
    meta = {
        "file": path.name, "sheet": sheet, "period": period, "rows": n, "lines": table.num_rows,
        "blank_rows": blank_rows,
        "blank_material": {m: {"lines": b["lines"], "value": round(b["value"], 2)}
                           for m, b in sorted(blank.items())},
        "mvt": {m or "": k for m, k in sorted(mvt.items(), key=lambda kv: kv[0] or "")},
        "returns": sum(mvt.get(m, 0) for m in C.RETURN_MVTS),
        "dates": dates, "value": round(value, 2), "qty": round(qty, 3),
        "secs": round(time.time() - t0, 1),
    }
    log(f"[purchase] GRN {path.name}: {n:,} lines of {period} in {meta['secs']}s")
    return table, meta


def _prune(cache, source: str, keep: Path) -> None:
    """Drop older cache entries of the same workbook (an edited or re-saved file)."""
    for side in C.cache_dir(cache).glob("grn_*.json"):
        if side.with_suffix(".parquet") == keep:
            continue
        try:
            same = json.loads(side.read_text()).get("source_path") == source
        except (OSError, ValueError):
            continue
        if same:
            side.with_suffix(".parquet").unlink(missing_ok=True)
            side.unlink(missing_ok=True)


def cached_report(path, cache=None) -> Optional[GrnFileReport]:
    """The cached report of one workbook as it is now (same path, size, mtime and reader
    version), or None when the cache does not hold it. Reads no workbook."""
    path = Path(path)
    out, side = _cache_paths(cache, _cache_key(path, path.stat()))
    if not (out.exists() and side.exists()):
        return None
    meta = json.loads(side.read_text())
    meta.pop("source_path", None)
    meta.pop("reader_version", None)
    return GrnFileReport(**{**meta, "cached": True})


def ensure_cached(path, cache=None, log=print) -> GrnFileReport:
    """Make sure one workbook's lines are in the cache; its report (cached=True when the
    cache already held this exact file)."""
    path = Path(path)
    rep = cached_report(path, cache)
    if rep is not None:
        return rep
    st = path.stat()
    out, side = _cache_paths(cache, _cache_key(path, st))
    table, meta = _read_workbook(path, log=log)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out)
    meta.update(source_size=st.st_size, source_mtime_ns=st.st_mtime_ns, cache_file=out.name)
    side_tmp = side.with_suffix(".json.part")
    side_tmp.write_text(json.dumps({**meta, "source_path": str(path.resolve()),
                                    "reader_version": C.GRN_READER_VERSION}, indent=1))
    os.replace(side_tmp, side)
    _prune(cache, str(path.resolve()), out)
    return GrnFileReport(**meta)


def _columns(extra: Sequence[str]) -> list:
    bad = [c for c in extra if c not in C.GRN_EXTRA_COLUMNS]
    if bad:
        raise ValueError(f"unknown extra GRN columns {bad}; choose from {list(C.GRN_EXTRA_COLUMNS)}")
    return ["period", *C.GRN_COLUMNS, *[c for c in C.GRN_EXTRA_COLUMNS if c in extra]]


def _load(rep: GrnFileReport, cache, columns: list) -> pd.DataFrame:
    path = C.cache_dir(cache) / rep.cache_file
    return pq.read_table(path, columns=columns).to_pandas(date_as_object=False)


def read_grn_file(path, cache=None, extra: Sequence[str] = (), log=print) -> tuple:
    """One GRN workbook -> (lines, GrnFileReport). Lines: one row per GRN line with a
    material; columns 'period' + config.GRN_COLUMNS (+ the extras asked for)."""
    rep = ensure_cached(path, cache=cache, log=log)
    return _load(rep, cache, _columns(extra)), rep


def read_grn_files(raw=None, cache=None, files=None, months=None, extra: Sequence[str] = (),
                   log=print) -> GrnRead:
    """Every GRN workbook (default: all of grn_files(raw)) -> GrnRead.

    Each workbook is assigned to the one month its GR dates fall in; two workbooks for one
    month abort. `months` limits the returned lines to those periods (each must exist).

    Lines: period (YYYY-MM), plant, sloc, material, material_desc, vendor_code, vendor_name,
    mvt_type ('101', '122', '161'), gr_date (datetime64, NaT when unreadable), gr_qty, uom,
    base_price, discount_value, net_price, total_wo_tax, tax_pct, unit_mrp, manufacturer,
    material_type, minor_category, major_group, minor_group, grn_formulary; codes are text,
    amounts float. Returns and lines outside any scope are included: see apply_scope."""
    folder = None
    if files is None:
        folder = find_grn_dir(raw)
        files = _excel_files(folder)
    files = [Path(p) for p in files]
    if not files:
        raise BuildError(f"no GRN workbooks in {folder}")
    columns = _columns(extra)
    reps, by_month = [], {}
    for p in files:
        rep = ensure_cached(p, cache=cache, log=log)
        if rep.period in by_month:
            raise BuildError(f"two GRN workbooks hold {rep.period}: {by_month[rep.period]!r} and "
                             f"{rep.file!r}")
        by_month[rep.period] = rep.file
        reps.append(rep)
    reps.sort(key=lambda r: r.period)
    if months:
        missing = sorted(set(months) - set(by_month))
        if missing:
            raise BuildError(f"no GRN workbook for {', '.join(missing)} (found: "
                             f"{', '.join(sorted(by_month))})")
        reps = [r for r in reps if r.period in set(months)]
    warnings = [f"{r.file}: {r.unparseable_dates:,} GR dates could not be read (kept, month "
                f"from the file)" for r in reps if r.unparseable_dates]
    frames = [_load(r, cache, columns) for r in reps]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return GrnRead(folder=folder, frame=frame, files=reps, warnings=warnings)


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
def _step_label(kind: str, values) -> str:
    v = ", ".join(sorted(values))
    return {"mvt_keep": f"movement type {v}",
            "series_keep": f"material series {v}",
            "sloc": f"not storage location {v}",
            "plant_prefix": "not plants " + ", ".join(f"{p}*" for p in sorted(values)),
            "plant": f"not plants {v}"}[kind]


def apply_scope(df: pd.DataFrame, scope: C.Scope, blank: Optional[pd.DataFrame] = None) -> tuple:
    """Keep the lines the scope counts -> (kept lines, funnel).

    Filters, in HCG's order: movement types kept (mvt_keep), material series kept (first
    digit, series_keep), storage locations dropped (sloc), plants dropped by prefix
    (plant_prefix) and by code (plant); a drop kind with no values is no step.

    funnel: one row per period x step: period, step (0 = every line read), kind, label,
    lines, value (Total Amount without tax) remaining after the step. `blank` is
    GrnRead.blank_lines(): lines the reader dropped for having no material still count in
    the steps before the series step (HCG's sheet filters movement type first)."""
    need = {"period", "mvt_type", "material", "sloc", "plant", "total_wo_tax"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"apply_scope needs the columns {sorted(missing)}")
    prefixed = (df["plant"].astype("str").str.startswith(scope.plant_prefix).fillna(False).astype(bool)
                if scope.plant_prefix else pd.Series(False, index=df.index))
    tests = {
        "mvt_keep": df["mvt_type"].isin(scope.mvt_keep),
        "series_keep": df["material"].astype("str").str[:1].isin(scope.series_keep),
        "sloc": ~df["sloc"].isin(scope.sloc),
        "plant_prefix": ~prefixed,
        "plant": ~df["plant"].isin(scope.plant),
    }
    b = blank if blank is not None else pd.DataFrame(columns=["period", "mvt_type", "lines", "value"])
    mask = pd.Series(True, index=df.index)
    steps = [("lines", "lines in the GRN", mask, b)]
    for kind in C.SCOPE_KINDS:
        values = scope.values(kind)
        if not values:
            continue
        mask = mask & tests[kind]
        if kind == "mvt_keep":
            b = b[b["mvt_type"].isin(values)]
        elif kind == "series_keep":                     # no material: in no series
            b = b.iloc[0:0]
        steps.append((kind, _step_label(kind, values), mask, b))
    rows = []
    value = df["total_wo_tax"].fillna(0.0)
    for step, (kind, label, m, b) in enumerate(steps):
        got = pd.DataFrame({"period": df["period"], "on": m, "value": value.where(m, 0.0)})
        g = got.groupby("period").agg(lines=("on", "sum"), value=("value", "sum"))
        g = g.add(b.groupby("period").agg(lines=("lines", "sum"), value=("value", "sum")),
                  fill_value=0)
        for period, r in g.iterrows():
            rows.append({"period": period, "step": step, "kind": kind, "label": label,
                         "lines": int(r["lines"]), "value": round(float(r["value"]), 2)})
    funnel = (pd.DataFrame(rows, columns=["period", "step", "kind", "label", "lines", "value"])
              .sort_values(["period", "step"], ignore_index=True))
    return df[mask].reset_index(drop=True), funnel
