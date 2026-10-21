"""Stream one month's consolidated sales sheet into the line-level cache.

FORMULARY_CACHE_DIR/lines_YYYY-MM.parquet (+ .json sidecar: source file, sheet, rows,
totals) holds one row per sales line with only the columns the formulary tables need.
It lives OUTSIDE the repo. Not kept, ever: UHID, PATNAME, IP_OPNUMBER, the full invoice
number (only its hospital prefix), BATCHCODE, ISSUECODE, EXPIRYDATE.
"""
from __future__ import annotations

import json
import operator
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.etl.formulary import config as C
from app.etl.formulary import excel as X
from app.etl.formulary.discover import BuildError

CONVERT = {
    "patient_type": X.upper,
    "hospital_name": X.text,
    "his_code": X.invoice_prefix,
    "sales_date": X.to_date,
    "bill_category": X.upper,
    "doctor_id": X.code,
    "doctor_name": X.name,
    "material": X.code,
    "material_desc": X.text,
    "brand_name": X.text,
    "generic_name": X.text,
    "qty": X.number,
    "total_mrp": X.money,
    "total_cost": X.number,
    "payer1": X.text,
    "manufacturer": X.name,
    "hcg_category": X.text,
    "hcg_payer": X.text,
    "hcg_segment": X.text,
}
_FLOAT = ("qty", "total_mrp", "total_cost")
SCHEMA = pa.schema(
    [("period", pa.string())]
    + [(c, pa.date32() if c == "sales_date" else pa.float64() if c in _FLOAT else pa.string())
       for c in CONVERT])


def cache_paths(cache, period: str) -> tuple:
    d = C.cache_dir(cache)
    return d / f"lines_{period}.parquet", d / f"lines_{period}.json"


def hcg_columns(hdr: list) -> dict:
    """Positions of HCG's three lookup columns in a normalised header (missing -> absent)."""
    out = {}
    for i, h in enumerate(hdr):
        if "hcg_category" not in out and C.HCG_CATEGORY_RE.match(h):
            out["hcg_category"] = i
        elif h == C.HCG_PAYER_HEADER and "hcg_payer" not in out:
            out["hcg_payer"] = i
        elif h == C.HCG_SEGMENT_HEADER and "hcg_segment" not in out:
            out["hcg_segment"] = i
    return out


def choose_sheet(wb: X.Workbook) -> tuple:
    """(sheet, header row index, header, has_hcg, warnings).

    The sheet must have every line-level column; among those, the one that also carries
    HCG's lookup columns wins (Jun 'IP&OP Sales Raw June 26 ', Jul 'IP&OP Jul 26 WS',
    Aug 'IP&OP Con WS August 26'). A future month without HCG's columns falls back to the
    line sheet and skips the agreement check, with a warning."""
    need = list(C.SALES_COLUMNS.values())
    full, plain = [], []
    for sh in wb.sheet_names:
        i, hdr = X.find_header(wb.rows(sh), need)
        if hdr is None:
            continue
        (full if len(hcg_columns(hdr)) == 3 else plain).append((sh, i, hdr))
    warnings = []
    if full:
        if len(full) > 1:
            warnings.append(f"several sheets carry HCG's columns {[s for s, _, _ in full]}; "
                            f"using {full[0][0]!r}")
        sh, i, hdr = full[0]
        return sh, i, hdr, True, warnings
    if plain:
        if len(plain) > 1:
            warnings.append(f"several line sheets {[s for s, _, _ in plain]}; using {plain[0][0]!r}")
        warnings.append(f"no sheet carries HCG's lookup columns; agreement check skipped")
        sh, i, hdr = plain[0]
        return sh, i, hdr, False, warnings
    raise BuildError(f"{wb.path.name}: no sheet has all of {', '.join(need)}")


def extract_month(sales_path, period: str, cache=None, limit: int = 0, log=print) -> dict:
    """Read the month's sheet into the cache and return the sidecar dict."""
    sales_path = Path(sales_path)
    out, side = cache_paths(cache, period)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    t0 = time.time()
    ym = (int(period[:4]), int(period[5:]))
    with X.Workbook(sales_path) as wb:
        sheet, hdr_row, hdr, has_hcg, warnings = choose_sheet(wb)
        log(f"[formulary] {period}: reading {sales_path.name!r} sheet {sheet!r}"
            f"{'' if has_hcg else ' (no HCG columns)'}")
        pos = {c: hdr.index(h) for c, h in C.SALES_COLUMNS.items()}
        pos.update(hcg_columns(hdr))
        read = [c for c in CONVERT if c in pos]
        absent = [c for c in CONVERT if c not in pos]
        width = max(pos.values()) + 1
        take = operator.itemgetter(*(pos[c] for c in read))
        buf: list = []
        st = {"rows": 0, "blank": 0, "cost": 0.0, "mrp": 0.0, "qty": 0.0,
              "no_cost": 0, "no_date": 0, "off_month": 0}
        types: Counter = Counter()
        writer = pq.ParquetWriter(str(tmp), SCHEMA, compression="zstd")
        try:
            def flush():
                # Convert a chunk column by column, each DISTINCT value once: a month has
                # ~880k lines but only 23 hospitals, ~30 dates, ~500 payers, ~10k items.
                if not buf:
                    return
                n = len(buf)
                data = {"period": [period] * n}
                for c, vals in zip(read, zip(*buf)):
                    fn = CONVERT[c]
                    memo = {v: fn(v) for v in set(vals)}
                    data[c] = [memo[v] for v in vals]
                for c in absent:
                    data[c] = [None] * n
                cost = data["total_cost"]
                st["no_cost"] += cost.count(None)
                st["cost"] += sum(v for v in cost if v is not None)
                st["mrp"] += sum(v for v in data["total_mrp"] if v is not None)
                st["qty"] += sum(v for v in data["qty"] if v is not None)
                for d, k in Counter(data["sales_date"]).items():
                    if d is None:
                        st["no_date"] += k
                    elif (d.year, d.month) != ym:
                        st["off_month"] += k
                types.update(data["patient_type"])
                writer.write_table(pa.table(data, schema=SCHEMA))
                st["rows"] += n
                buf.clear()

            it = wb.rows(sheet)
            for _ in range(hdr_row + 1):
                next(it)
            for r in it:
                if r is None:
                    st["blank"] += 1
                    continue
                if len(r) < width:
                    r = list(r) + [None] * (width - len(r))
                t = take(r)
                if t.count(None) + t.count("") == len(t):
                    st["blank"] += 1
                    continue
                buf.append(t)
                if limit and st["rows"] + len(buf) >= limit:
                    break
                if len(buf) >= C.CHUNK_ROWS:
                    flush()
                    log(f"[formulary] {period}: {st['rows']:,} rows {time.time() - t0:.0f}s")
            flush()
        except BaseException:
            writer.close()
            tmp.unlink(missing_ok=True)
            raise
        writer.close()
    rows, blank = st["rows"], st["blank"]
    cost, mrp, qty = st["cost"], st["mrp"], st["qty"]
    no_cost, no_date, off_month = st["no_cost"], st["no_date"], st["off_month"]
    if rows == 0:
        tmp.unlink(missing_ok=True)
        raise BuildError(f"{sales_path.name} sheet {sheet!r} read 0 rows — a reader failure, "
                         f"not an empty month. Nothing was written.")
    os.replace(tmp, out)
    meta = {
        "period": period,
        "source_file": sales_path.name,
        "source_size": sales_path.stat().st_size,
        "sheet": sheet,
        "has_hcg_columns": has_hcg,
        "rows": rows,
        "blank_rows_skipped": blank,
        "limited": bool(limit),
        "cost": round(cost, 2),
        "mrp": round(mrp, 2),
        "qty": qty,
        "cost_empty_lines": no_cost,
        "dates_missing": no_date,
        "dates_outside_month": off_month,
        "patient_types": {str(k): v for k, v in sorted(types.items(), key=lambda kv: str(kv[0]))},
        "warnings": warnings,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "secs": round(time.time() - t0, 1),
    }
    side_tmp = side.with_suffix(".json.part")
    side_tmp.write_text(json.dumps(meta, indent=1))
    os.replace(side_tmp, side)
    log(f"[formulary] {period}: {rows:,} rows, cost {cost:,.2f} in {meta['secs']}s")
    return meta


def read_sidecar(cache, period: str) -> dict:
    out, side = cache_paths(cache, period)
    if not out.exists() or not side.exists():
        raise BuildError(f"no cached lines for {period} in {out.parent} (run without --from-cache)")
    meta = json.loads(side.read_text())
    if meta.get("limited"):
        raise BuildError(f"the cached lines for {period} come from a --limit run; rebuild them")
    return meta


def cached_periods(cache=None) -> list:
    d = C.cache_dir(cache)
    if not d.is_dir():
        return []
    return sorted(p.stem[len("lines_"):] for p in d.glob("lines_*.parquet")
                  if (d / f"{p.stem}.json").exists())
