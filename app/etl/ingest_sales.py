"""Streaming ETL for IP + OP Sales (billing) → small aggregate parquet tables.

The raw sales files are large (IP xlsx up to ~280MB) and have inconsistent schemas
month to month (29 vs 41 cols, different sheet names, .xlsx and .xlsb). We stream
each file row-by-row (never hold a whole file in memory), auto-detect the data
sheet + header, harmonise columns, keep the months of LEGACY_SALES_MONTHS (Dec-2025..May-2026,
on purpose: see below), and accumulate aggregates the dashboards need:

  sales_monthly.parquet        month × patient(IP/OP) → revenue, cost, qty, lines
  sales_by_hospital.parquet    hospital(code) → revenue, cost, qty
  sales_by_manufacturer.parquet manufacturer → revenue, cost, qty
  sales_by_material.parquet    material → desc, group, revenue, cost, qty
  sales_by_material_month.parquet  material × period → revenue, cost, qty, lines
  sales_totals.parquet         patient → revenue, cost, qty, lines

Revenue = TOTALMRP (billed), Cost = TOTALCOSTPRICE, Margin = revenue − cost (real).

Run from the repo root, pointing SALES_RAW at the folder that holds "IP Sales" and
"OP Sales" (see RUNBOOK_HCG.md):
    SALES_RAW=/path/to/folder python -m app.etl.ingest_sales
SALES_MONTHS=YYYY-MM..YYYY-MM moves the window it keeps (default LEGACY_SALES_WINDOW).
"""
from __future__ import annotations
import os, glob, re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd
from openpyxl import load_workbook

try:
    from pyxlsb import open_workbook as open_xlsb
except Exception:
    open_xlsb = None

from app.core.config import settings
# One rule for material codes across the ETLs (the formulary and spend builds use it too).
from app.etl.formulary.classify import norm_code

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend root
# SALES_RAW wins; otherwise the same raw-data root the rest of the ETL uses. The old
# default was one developer's Mac path, which silently read nothing anywhere else.
RAW = os.environ.get("SALES_RAW") or settings.RAW_DATA_DIR
KPI = os.path.join(HERE, "app", "data", "kpi")
LIMIT = int(os.environ.get("SALES_LIMIT", "0"))  # >0 = only first N data rows/file (prototype)

# ── The months this ingest keeps ──────────────────────────────────────────────────────────
# Dec-2025..May-2026, on purpose. These legacy sales_*.parquet tables sit beside the stock count
# of 31-May-26 and the internal issues up to it: days on hand, turnover, non-moving and the
# billed-vs-internal views set them side by side. Every month from Jun-26 on is served by the
# formulary build (its line cache and sales_items, app/etl/formulary), never by this ingest, so a
# re-run on a newer HIS file still keeps these months only, and its log says how many rows it
# left out. SALES_MONTHS=YYYY-MM..YYYY-MM moves the window without a code change.
# Nothing reads the window from here: app.core.data_access takes it back from the published
# sales_monthly (legacy_sales_months), so the tables and every rate built on them (billed units
# per day, turnover per year) always agree, and GET /meta/windows reports it as `legacy_sales`.
LEGACY_SALES_WINDOW = "2025-12..2026-05"
_YM = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def window_months(spec: str) -> tuple:
    """'2025-12..2026-05' -> ('2025-12', '2026-01', ..., '2026-05'): every month from the first to
    the last, both included. A spec that is not two YYYY-MM months in order stops the run."""
    parts = [p.strip() for p in str(spec).split("..")]
    if len(parts) != 2 or not all(_YM.match(p) for p in parts) or parts[0] > parts[1]:
        raise SystemExit(f"[sales] ABORT — the sales window {spec!r} is not YYYY-MM..YYYY-MM "
                         "(first month first). Nothing was written.")
    out, (y, m) = [], (int(parts[0][:4]), int(parts[0][5:]))
    while f"{y:04d}-{m:02d}" <= parts[1]:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return tuple(out)


# The window as months ('2025-12', ...) and as the (year, month) keys the row loop checks.
LEGACY_SALES_MONTHS = window_months(os.environ.get("SALES_MONTHS") or LEGACY_SALES_WINDOW)
KEEP = {(int(m[:4]), int(m[5:])) for m in LEGACY_SALES_MONTHS}
EXCEL_BASE = datetime(1899, 12, 30)
# A file whose SALESDATE cannot be read on more than this share of its rows stops the run: that is
# a reader fault (a date format parse_ym does not know, a blank column), not rows outside the
# window, and leaving them out would drop the month as surely as a file that reads empty. A few
# blank dates (a total row, an empty line) stay far below it and are only counted.
MAX_NO_DATE_SHARE = 0.5

FIELDS = {
    "date": ["SALESDATE", "BILLDATE"],
    "mat": ["MATERIALCODE", "ITEMCODE"],
    "desc": ["MATERIALLINEITEM", "ITEMNAME"],
    "qty": ["QTY"],
    "rev": ["TOTALMRP"],
    "cost": ["TOTALCOSTPRICE"],
    "mfr": ["MANUFACTURERNAME"],
    "inv": ["SALESINVOICENO", "BILLNO"],
    "loc": ["LOCATIONNAME", "STORENAME"],
}
REQUIRED = ("date", "qty", "rev", "cost", "mat")


def parse_ym(v):
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return (v.year, v.month)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            d = EXCEL_BASE + timedelta(days=float(v)); return (d.year, d.month)
        except Exception:
            return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            d = datetime.strptime(s[:19] if len(s) >= 19 and " " in s else s.split(" ")[0], fmt)
            return (d.year, d.month)
        except Exception:
            pass
    try:
        d = EXCEL_BASE + timedelta(days=float(s)); return (d.year, d.month)
    except Exception:
        return None


def outside_window(ym) -> str:
    """Why a row the window does not keep is left out: 'no_date' when its SALESDATE is blank or
    unreadable (parse_ym gave None), else 'before' or 'after' the window. Only 'after' is served
    elsewhere (the formulary build's sales_items)."""
    if ym is None:
        return "no_date"
    return "before" if ym < min(KEEP) else "after"


def left_out_lines(left: Counter) -> list:
    """The run log's lines for the rows the window left out, by why (outside_window)."""
    first, last = LEGACY_SALES_MONTHS[0], LEGACY_SALES_MONTHS[-1]
    out = []
    if left["before"]:
        out.append(f"[sales] {left['before']:,} rows dated before {first} were left out "
                   "(SALES_MONTHS moves the window).")
    if left["after"]:
        out.append(f"[sales] {left['after']:,} rows dated after {last} were left out; months after the "
                   "window come from the formulary build (sales_items).")
    if left["no_date"]:
        out.append(f"[sales] {left['no_date']:,} rows have no readable SALESDATE (blank, or a date format "
                   "the reader does not know) and were not counted.")
    return out


def col_map(header):
    H = [str(h).strip().upper() if h is not None else "" for h in header]
    idx = {}
    for f, cands in FIELDS.items():
        for c in cands:
            if c in H:
                idx[f] = H.index(c); break
    return idx if all(k in idx for k in REQUIRED) else None


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def iter_xlsx(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        for sh in wb.sheetnames:
            ws = wb[sh]
            it = ws.iter_rows(values_only=True)
            idx = None
            for _ in range(8):
                try:
                    row = next(it)
                except StopIteration:
                    break
                idx = col_map(row)
                if idx:
                    break
            if not idx:
                continue
            for row in it:
                yield idx, row
            return
    finally:
        wb.close()


def iter_xlsb(path):
    if open_xlsb is None:
        return
    with open_xlsb(path) as wb:
        for sh in wb.sheets:
            with wb.get_sheet(sh) as sheet:
                idx = None
                for r in sheet.rows():
                    vals = [c.v for c in r]
                    if idx is None:
                        idx = col_map(vals)
                        continue
                    yield idx, vals
                if idx is not None:
                    return


def records(path):
    it = iter_xlsb(path) if path.lower().endswith(".xlsb") else iter_xlsx(path)
    n = 0
    for idx, row in it:
        def g(f):
            i = idx.get(f)
            return row[i] if (i is not None and i < len(row)) else None
        yield g
        n += 1
        if LIMIT and n >= LIMIT:
            return


def hosp_of(inv, loc):
    if inv:
        s = str(inv)
        if "-" in s:
            return s.split("-")[0].strip().upper()
    return str(loc).strip() if loc else "UNKNOWN"


_FLOAT_CODE = re.compile(r"\d+\.0+")


def float_codes(codes) -> list:
    """Material codes still written the way str() writes a float ('101313.0')."""
    return [c for c in codes if _FLOAT_CODE.fullmatch(str(c))]


def run():
    missing = [sub for sub in ("IP Sales", "OP Sales") if not os.path.isdir(os.path.join(RAW, sub))]
    if missing:
        # Without this a wrong path reads zero files, and the zero-row guard below never
        # fires because it only checks files that were found.
        raise SystemExit(
            f"[sales] ABORT — {RAW!r} has no {' / '.join(missing)} folder. Set SALES_RAW to the "
            "folder that holds 'IP Sales' and 'OP Sales'. Nothing was written.")
    tot = defaultdict(lambda: defaultdict(float))
    month = defaultdict(lambda: defaultdict(float))
    hosp = defaultdict(lambda: defaultdict(float))
    mfr = defaultdict(lambda: defaultdict(float))
    mat = defaultdict(lambda: defaultdict(float))
    mfr_mat = defaultdict(lambda: defaultdict(float))   # (manufacturer, material) → for mfr drill
    hosp_mat = defaultdict(lambda: defaultdict(float))  # (hospital, material) → for hospital drill
    # (material, month) — the one cut nobody produced, though every raw billing row carries
    # both SALESDATE and MATERIALCODE. Its absence is why "show me the sales trend of
    # KEYTRUDA" was answerable only as a six-month total: sales_monthly has the month and no
    # material, sales_by_material has the material and no month, and the cross was never
    # written. The assistant then told users, correctly about the WAREHOUSE and wrongly
    # about the DATA, that no month-by-month figure existed for an item.
    mat_month = defaultdict(lambda: defaultdict(float))
    read_counts: dict = {}          # path -> rows SEEN, so a silent reader failure is loud
    matdesc = {}

    def files(sub):
        d = os.path.join(RAW, sub)
        return sorted(glob.glob(d + "/*.xlsx") + glob.glob(d + "/*.xlsb"))

    sources = [("IP", f) for f in files("IP Sales")] + [("OP", f) for f in files("OP Sales")]
    print(f"[sales] keeping {LEGACY_SALES_MONTHS[0]}..{LEGACY_SALES_MONTHS[-1]} "
          f"({len(LEGACY_SALES_MONTHS)} months, LEGACY_SALES_MONTHS)", flush=True)
    left_total = Counter()
    no_date = {}                    # path -> rows whose SALESDATE could not be read
    for patient, path in sources:
        kept = 0; seen = 0; left = Counter()
        for g in records(path):
            seen += 1
            ym = parse_ym(g("date"))
            if ym not in KEEP:
                left[outside_window(ym)] += 1
                continue
            rev = _num(g("rev")); cost = _num(g("cost")); qty = _num(g("qty"))
            if rev is None:
                continue
            cost = cost or 0.0; qty = qty or 0.0
            # pyxlsb returns every numeric cell as a float and openpyxl returns whole
            # numbers as int, so str() keyed the four .xlsb OP months as '101313.0' and the
            # .xlsx months as '101313'. Every material-keyed table then held each product
            # twice, and the '.0' half never matched dim_material, so it had no group.
            code = norm_code(g("mat"))
            if not code or code.lower() == "none":
                continue
            ymk = f"{ym[0]:04d}-{ym[1]:02d}"
            hk = hosp_of(g("inv"), g("loc"))
            for bucket, key in ((tot, patient), (month, (patient, ymk)),
                                (hosp, hk), (mat, code)):
                bucket[key]["revenue"] += rev; bucket[key]["cost"] += cost
                bucket[key]["qty"] += qty; bucket[key]["lines"] += 1
            hm = hosp_mat[(hk, code)]
            hm["revenue"] += rev; hm["cost"] += cost; hm["qty"] += qty; hm["lines"] += 1
            mmo = mat_month[(code, ymk)]
            mmo["revenue"] += rev; mmo["cost"] += cost; mmo["qty"] += qty; mmo["lines"] += 1
            m = g("mfr")
            if m and str(m).strip().lower() not in ("none", "nan", ""):
                mt = str(m).strip().title()
                mm = mfr[mt]
                mm["revenue"] += rev; mm["cost"] += cost; mm["qty"] += qty; mm["lines"] += 1
                mmx = mfr_mat[(mt, code)]
                mmx["revenue"] += rev; mmx["cost"] += cost; mmx["qty"] += qty; mmx["lines"] += 1
            matdesc.setdefault(code, str(g("desc")) if g("desc") else code)
            kept += 1
        read_counts[path] = seen
        no_date[path] = left["no_date"]
        left_total += left
        print(f"[sales] {patient:2s} {os.path.basename(path)[:38]:38s} seen={seen:>8,} kept={kept:>8,} "
              f"outside_window={left['before'] + left['after']:>8,} no_date={left['no_date']:>8,}", flush=True)
    # Said out loud, each for what it is: a newer HIS file's later months are left out by design
    # (the formulary build serves them); an unreadable date is not a month served elsewhere.
    for line in left_out_lines(left_total):
        print(line, flush=True)

    # category from material master
    grp = {}
    try:
        dm = pd.read_parquet(os.path.join(KPI, "..", "curated", "dim_material.parquet"))
    except Exception:
        dm = None
    if dm is None:
        try:
            dm = pd.read_parquet(os.path.join(HERE, "app", "data", "curated", "dim_material.parquet"))
        except Exception:
            dm = None
    if dm is not None and "material" in dm and "material_group" in dm:
        grp = dict(zip(dm["material"].astype(str), dm["material_group"].astype(str)))

    # REFUSE TO WRITE A PARTIAL DATASET.
    #
    # pyxlsb is an optional import guarded by try/except. With it missing, all four .xlsb
    # OP-sales months read ZERO rows, the run "succeeded", and it overwrote seven parquets
    # with ₹492.40 Cr in place of ₹521.67 Cr — 4 months of outpatient billing gone, and
    # nothing in the log said anything louder than "seen=0 kept=0". Every dashboard and
    # every AI answer would have quietly moved by ₹29 Cr.
    #
    # A file that yields no rows at all is a reader failure, not an empty month. Say so and
    # stop, rather than half-writing the warehouse.
    empty = [os.path.basename(p_) for p_, n_ in read_counts.items() if n_ == 0]
    if empty:
        raise SystemExit(
            "[sales] ABORT — these files produced ZERO rows, so the aggregate would be "
            "incomplete:\n  " + "\n  ".join(empty) +
            "\n\nNothing was written. A .xlsb file reading empty usually means pyxlsb is "
            "not installed (pip install pyxlsb). Fix the reader and re-run; do NOT ship "
            "these aggregates, because they silently drop whole months.")

    # Same idea for the dates: a file whose SALESDATE mostly cannot be read drops its month too.
    undated = [f"{os.path.basename(p_)}: {no_date[p_]:,} of {n_:,} rows" for p_, n_ in read_counts.items()
               if no_date[p_] > MAX_NO_DATE_SHARE * n_]
    if undated:
        raise SystemExit(
            "[sales] ABORT — these files have no readable SALESDATE on most rows, so their months "
            "would be left out:\n  " + "\n  ".join(undated) +
            "\n\nNothing was written. Check the date column's format (parse_ym) and re-run.")

    # Same idea for the material key: if a reader change ever brings back float-shaped
    # codes, stop before they split every product in two again.
    floaty = float_codes(mat)
    if floaty:
        raise SystemExit(
            f"[sales] ABORT — {len(floaty):,} material codes are float-formatted, e.g. "
            f"{floaty[:3]}. Nothing was written; normalise the codes (norm_code) and re-run.")

    os.makedirs(KPI, exist_ok=True)
    pd.DataFrame([{"patient": k, **v} for k, v in tot.items()]).to_parquet(os.path.join(KPI, "sales_totals.parquet"), index=False)
    pd.DataFrame([{"patient": k[0], "month": k[1], **v} for k, v in month.items()]).to_parquet(os.path.join(KPI, "sales_monthly.parquet"), index=False)
    pd.DataFrame([{"hospital": k, **v} for k, v in hosp.items()]).to_parquet(os.path.join(KPI, "sales_by_hospital.parquet"), index=False)
    pd.DataFrame([{"manufacturer": k, **v} for k, v in mfr.items()]).to_parquet(os.path.join(KPI, "sales_by_manufacturer.parquet"), index=False)
    pd.DataFrame([{"material": k, "desc": matdesc.get(k, k), "group": grp.get(k, ""), **v} for k, v in mat.items()]).to_parquet(os.path.join(KPI, "sales_by_material.parquet"), index=False)
    pd.DataFrame([{"manufacturer": k[0], "material": k[1], "desc": matdesc.get(k[1], k[1]), "group": grp.get(k[1], ""), **v} for k, v in mfr_mat.items()]).to_parquet(os.path.join(KPI, "sales_by_material_mfr.parquet"), index=False)
    pd.DataFrame([{"hospital": k[0], "material": k[1], "desc": matdesc.get(k[1], k[1]), "group": grp.get(k[1], ""), **v} for k, v in hosp_mat.items()]).to_parquet(os.path.join(KPI, "sales_by_material_hospital.parquet"), index=False)
    # material x month. `period` is named to match every other month-grained table so it
    # sorts chronologically ('2025-12' < '2026-01') rather than by calendar month number.
    pd.DataFrame([{"material": k[0], "period": k[1], "desc": matdesc.get(k[0], k[0]),
                   "group": grp.get(k[0], ""), **v} for k, v in mat_month.items()]
                 ).to_parquet(os.path.join(KPI, "sales_by_material_month.parquet"), index=False)
    print(f"[sales] wrote mfr_mat={len(mfr_mat):,} hosp_mat={len(hosp_mat):,} pairs", flush=True)

    grand = sum(v["revenue"] for v in tot.values())
    gcost = sum(v["cost"] for v in tot.values())
    print(f"\n[sales] DONE. revenue=₹{grand/1e7:.1f}Cr cost=₹{gcost/1e7:.1f}Cr margin=₹{(grand-gcost)/1e7:.1f}Cr ({(grand-gcost)/grand*100:.1f}%)")
    print(f"[sales] patients: " + ", ".join(f"{k}=₹{v['revenue']/1e7:.1f}Cr" for k, v in tot.items()))
    print(f"[sales] hospitals={len(hosp)} manufacturers={len(mfr)} materials={len(mat)}")


if __name__ == "__main__":
    run()
